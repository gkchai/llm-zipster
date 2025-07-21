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

text = ("NYC is a melting pot of cultures, with diverse neighborhoods like Chinatown, Little Italy, and Harlem "
        "each offering unique flavors, traditions, and artistic expressions. The city's theater district, "
        "Broadway, is a mecca for performing arts, drawing millions with its world-class productions. Museums "
        "like the Metropolitan Museum of Art and the Museum of Modern Art house invaluable collections spanning "
        "centuries and continents.")
data = text.encode("utf-8")
zipster = LLMZipster(verbose=True, num_bits=512)
compressed = zipster.compress(data)
decompressed = zipster.decompress(compressed)
```