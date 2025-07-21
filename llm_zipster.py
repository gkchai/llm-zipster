"""
Arithmetic coding/decoding with llm next-token probabilities.
"""

from decimal import Decimal, getcontext
from typing import Callable, List, Optional
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import math
import base64
import argparse
import sys

getcontext().prec = 500  # Set decimal precision (increase for longer texts)


def load_model_and_tokenizer(model_name: str ="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, resume_download=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def get_next_token_probs(prompt: str, tokenizer, model, device) -> np.ndarray:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


class ArithmeticCoder:
    def __init__(self, tokenizer, probs_fn: Callable[[str], np.ndarray], verbose: bool = False):
        self.tokenizer = tokenizer
        self.probs_fn = probs_fn
        self.verbose = verbose

    def encode(self, tokens: List[int]) -> Decimal:
        """Encode a sequence of tokens into a Decimal using arithmetic coding."""
        low = Decimal(0)
        high = Decimal(1)
        progress_bar = tqdm(
            total=len(tokens),
            mininterval=1 / 30,
            desc="Compressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
            disable=not self.verbose,
        )
        for i, token in enumerate(tokens):
            prompt = self.tokenizer.decode(tokens[:i]) if i > 0 else ' '
            probs = self.probs_fn(prompt)
            cum_probs = np.cumsum(probs)
            new_low = low + (high - low) * Decimal(float(cum_probs[token - 1])) if token > 0 else low
            new_high = low + (high - low) * Decimal(float(cum_probs[token]))
            low, high = new_low, new_high
            progress_bar.update(1)
            if high == low:
                raise ValueError(
                    f"Decimal precision too low: interval collapsed at token {i}. "
                    f"Try increasing decimal precision with getcontext().prec (current: {getcontext().prec})."
                )
        progress_bar.close()
        return (low + high) / 2

    def decode(self, encoded_value: Decimal, seq_length: int) -> List[int]:
        """Decode a Decimal value into a sequence of tokens using arithmetic decoding."""
        low = Decimal(0)
        high = Decimal(1)
        tokens = []
        progress_bar = tqdm(
            total=seq_length,
            mininterval=1 / 30,
            desc="Decompressing",
            unit="tok",
            leave=False,
            dynamic_ncols=True,
            disable=not self.verbose,
        )
        for _ in range(seq_length):
            prompt = self.tokenizer.decode(tokens) if tokens else ' '
            probs = self.probs_fn(prompt)
            cum_probs = np.cumsum(probs)
            scaled_value = float((encoded_value - low) / (high - low))
            token = int(np.searchsorted(cum_probs, scaled_value, side='right'))
            new_low = low + (high - low) * Decimal(float(cum_probs[token - 1])) if token > 0 else low
            new_high = low + (high - low) * Decimal(float(cum_probs[token]))
            low, high = new_low, new_high
            tokens.append(token)
            progress_bar.update(1)
        progress_bar.close()
        return tokens


def decimal_fraction_to_bytes(value: Decimal, num_bits: int) -> bytes:
    """
    Convert the fractional part of a Decimal (0 <= value < 1) to bytes using num_bits of precision.
    """
    assert 0 <= value < 1, "Value must be in [0, 1)"
    scaled = int((value * (1 << num_bits)).to_integral_value())
    num_bytes = math.ceil(num_bits / 8)
    return scaled.to_bytes(num_bytes, byteorder='big')


def bytes_to_decimal_fraction(byte_data: bytes, num_bits: int) -> Decimal:
    """
    Convert bytes (representing a fractional part) back to a Decimal in [0, 1).
    """
    as_int = int.from_bytes(byte_data, byteorder='big')
    return Decimal(as_int) / Decimal(1 << num_bits)


class LLMZipster:
    def __init__(self, verbose=True, model_name="gpt2", num_bits=512):
        self.tokenizer, self.model, self.device = load_model_and_tokenizer(model_name)
        self.verbose = verbose
        self.num_bits = num_bits

        def probs_fn(prompt):
            return get_next_token_probs(prompt, self.tokenizer, self.model, self.device)

        self.coder = ArithmeticCoder(self.tokenizer, probs_fn, verbose=verbose)

    def compress(self, uncompressed: bytes) -> bytes:
        text = uncompressed.decode("utf-8")
        print(f"Original text: {text}")
        print(f"Original size: {len(uncompressed)} bytes")
        tokens = self.tokenizer.encode(text)
        encoded_value = self.coder.encode(tokens)
        encoded_bytes = decimal_fraction_to_bytes(encoded_value, self.num_bits)
        b64_bytes = base64.b64encode(encoded_bytes)
        num_tokens_bytes = len(tokens).to_bytes(4, byteorder='big')
        compressed = num_tokens_bytes + b64_bytes
        print(f"Compressed bytes (base64): {compressed}")
        print(f"Compressed size: {len(compressed)} bytes")
        compression_ratio = len(uncompressed) / len(compressed) if len(compressed) > 0 else 0
        print(f"Compression ratio (original/compressed): {compression_ratio:.3f}")
        return compressed

    def decompress(self, compressed: bytes) -> bytes:
        num_tokens = int.from_bytes(compressed[:4], byteorder='big')
        b64_bytes = compressed[4:]
        encoded_bytes = base64.b64decode(b64_bytes)
        encoded_value = bytes_to_decimal_fraction(encoded_bytes, self.num_bits)
        tokens = self.coder.decode(encoded_value, num_tokens)
        text = self.tokenizer.decode(tokens)
        decompressed = text.encode("utf-8")
        print(f"Decompressed text: {text}")
        return decompressed


def main():
    text = ("NYC is a melting pot of cultures, with diverse neighborhoods like Chinatown, Little Italy, and Harlem "
            "each offering unique flavors, traditions, and artistic expressions. The city's theater district, "
            "Broadway, is a mecca for performing arts, drawing millions with its world-class productions. Museums "
            "like the Metropolitan Museum of Art and the Museum of Modern Art house invaluable collections spanning "
            "centuries and continents.")
    data = text.encode("utf-8")
    zipster = LLMZipster(verbose=True, num_bits=1024)
    compressed = zipster.compress(data)
    decompressed = zipster.decompress(compressed)
    if decompressed != data:
        print("Decompressed data does not match original!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMZipster: Compress or decompress text using GPT-2 arithmetic coding.")
    parser.add_argument('--compress', action='store_true', help='Compress input file or --text and write to output file')
    parser.add_argument('--decompress', action='store_true', help='Decompress base64-encoded input file and write to output file or stdout')
    parser.add_argument('--input', '-i', type=str, help='Input file path (for compress or decompress)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--text', type=str, help='Text to compress (overrides --input if given)')
    parser.add_argument('--num-bits', type=int, default=512, help='Number of bits for arithmetic coding (default: 512)')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bar')
    args = parser.parse_args()

    verbose = not args.no_progress

    if args.compress:
        if args.text is not None:
            data = args.text.encode('utf-8')
        else:
            if not args.input or not args.output:
                print('Compression requires --input and --output (or --text and --output).', file=sys.stderr)
                sys.exit(1)
            with open(args.input, 'rb') as f:
                data = f.read()
        zipster = LLMZipster(verbose=verbose, num_bits=args.num_bits)
        compressed = zipster.compress(data)
        with open(args.output, 'wb') as f:
            f.write(compressed)
    elif args.decompress:
        if not args.input:
            print('Decompression requires --input.', file=sys.stderr)
            sys.exit(1)
        with open(args.input, 'rb') as f:
            b64_data = f.read()
        zipster = LLMZipster(verbose=verbose, num_bits=args.num_bits)
        decompressed = zipster.decompress(b64_data)
        if args.output:
            with open(args.output, 'wb') as f:
                f.write(decompressed)
    else:
        main()