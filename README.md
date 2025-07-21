# LLMZipster

LLMZipster is a compressor and decompressor utility that uses arithmetic coding with LLM next token probabilities to compress and decompress text. 

## Usage

### Command Line

#### Compress a file:
```bash
python llm_zipster.py --compress --input input.txt --output compressed.b64
```

#### Compress a string:
```bash
python llm_zipster.py --compress --text "Hello, world!" --output compressed.b64
```

#### Decompress to a file:
```bash
python llm_zipster.py --decompress --input compressed.b64 --output output.txt
```

#### Decompress and print to stdout:
```bash
python llm_zipster.py --decompress --input compressed.b64
```

#### Additional options:
- `--num-bits N` : Set the number of bits for arithmetic coding (default: 512)
- `--no-progress` : Hide the progress bar

### Python API

```python
from llm_zipster import LLMZipster

zipster = LLMZipster(verbose=True, num_bits=512)
with open('input.txt', 'rb') as f:
    data = f.read()
compressed = zipster.compress(data)
with open('compressed.b64', 'wb') as f:
    f.write(compressed)

# Decompress
with open('compressed.b64', 'rb') as f:
    compressed = f.read()
decompressed = zipster.decompress(compressed)
with open('output.txt', 'wb') as f:
    f.write(decompressed)
```

## License

MIT License