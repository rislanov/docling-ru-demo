# Docling RU Demo

Simple Python script for testing and debugging PDF file recognition in Russian language with complex structures using [Docling](https://github.com/DS4SD/docling) from IBM.

## Features

✓ **PDF to Markdown conversion** - high-quality document transformation  
✓ **Russian language support** - built-in OCR support for Russian text  
✓ **Complex structures** - tables, images, formulas, code  
✓ **Apple Silicon optimization** - automatic GPU usage on M1/M2/M3 chips  
✓ **OCR for scanned documents** - text recognition from images  

## Requirements

- Python 3.8 or higher
- macOS 12.3+ (for Apple Silicon GPU support) or Linux/Windows

## Installation

### Automatic installation (recommended)

```bash
git clone https://github.com/rislanov/docling-ru-demo.git
cd docling-ru-demo
./install.sh
```

The installation script will create a virtual environment in the `venv` directory and install all dependencies there.

After installation, activate the virtual environment:
```bash
source venv/bin/activate
```

### Manual installation

1. Clone the repository:
```bash
git clone https://github.com/rislanov/docling-ru-demo.git
cd docling-ru-demo
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Check installation:
```bash
python check_deps.py
```

**Note:** First run may take time as Docling will download required AI models (~500MB).

## Usage

**Important:** Make sure to activate the virtual environment before running the scripts:
```bash
source venv/bin/activate
```

### Basic usage

Convert PDF file to Markdown:
```bash
python pdf_to_md.py document.pdf
```

Result will be saved to `document.md` in the same directory.

### Specify output file

```bash
python pdf_to_md.py document.pdf -o output.md
```

### Full paths

```bash
python pdf_to_md.py /path/to/document.pdf -o /path/to/output.md
```

### Help

```bash
python pdf_to_md.py --help
```

## Usage Examples

```bash
$ python pdf_to_md.py example.pdf
============================================================
Input file: /Users/user/documents/example.pdf
Output file: /Users/user/documents/example.md
============================================================

✓ Using Apple Silicon GPU (Metal Performance Shaders)
Starting PDF processing...
This may take some time depending on document size and complexity...

✓ Successfully converted!
✓ Result saved to: /Users/user/documents/example.md

Statistics:
  - Output file size: 15.42 KB
  - Character count: 15789
```

## Apple Silicon (M1/M2/M3) Support

The script automatically detects Apple Silicon GPU availability and uses Metal Performance Shaders (MPS) to accelerate processing. This provides:

- Significant speedup compared to CPU processing
- Efficient use of unified memory
- Support for large documents

For optimal performance, make sure the latest version of PyTorch with MPS support is installed.

## Architecture

The script uses the following components:

1. **Docling** - main engine for document conversion
2. **PyTorch** - for AI models for structure recognition
3. **OCR backend (EasyOCR)** - for Russian text recognition
4. **MPS backend** - for acceleration on Apple Silicon

## Troubleshooting

### Installation issues

If you encounter problems installing dependencies:

```bash
# Update pip
pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt --no-cache-dir
```

### Memory issues

For very large PDF files, more memory may be required. On Apple Silicon you can use:

```bash
# Increase memory limit for Python
ulimit -s 65536
python pdf_to_md.py large_document.pdf
```

### OCR not working with Russian language

Make sure all Docling dependencies are installed. On first run, the library will automatically download required language models.

## Project Structure

```
docling-ru-demo/
├── pdf_to_md.py        # Main conversion script
├── check_deps.py       # Dependency check script
├── install.sh          # Automatic installation script
├── requirements.txt    # Python dependencies list
├── README.md           # Documentation (this file)
└── .gitignore          # Ignored files
```

## Technical Details

### Supported input formats
- PDF (including scanned documents)
- Support for various text encodings

### Output format
- Markdown (.md)
- Preserves document structure
- Support for tables, lists, headings
- Image extraction as links

### Performance

Processing time depends on:
- Document size
- Structure complexity
- Presence of scanned pages (OCR is slower)
- Hardware used (CPU vs GPU)

Approximate time for a 10-page document:
- CPU: 30-60 seconds
- Apple Silicon GPU (MPS): 10-20 seconds

## License

This project uses Docling from IBM, which is distributed under the MIT license.

## Links

- [Docling GitHub](https://github.com/DS4SD/docling)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [PyTorch MPS Support](https://pytorch.org/docs/stable/notes/mps.html)

## Author

Demonstration project for testing Docling capabilities with Russian language documents.