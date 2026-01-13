#!/usr/bin/env python3
"""
Script for testing and debugging PDF file recognition in Russian language
using Docling from IBM.

Supports:
- Complex document structures
- Tables, images, formulas
- Apple M-series chips (GPU via MPS backend)
- Russian language (built-in OCR support)
"""

import argparse
import sys
import signal
import logging
import os
import time
from pathlib import Path

# Enable display of HuggingFace model loading progress
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # Stable download
os.environ.setdefault("TQDM_DISABLE", "0")  # Enable progress bars

import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# Configure logging to display model loading progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Increase log verbosity for debugging initialization issues
logger = logging.getLogger()


def set_verbose_logging():
    """Enables detailed logging for debugging."""
    logger.setLevel(logging.INFO)
    # Add logging for important modules
    logging.getLogger('docling').setLevel(logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)  # Reduce noise from HF
    

set_verbose_logging()


def signal_handler(sig, frame):
    """Ctrl+C interrupt handler."""
    print("\n\n⚠ Interrupted by user (Ctrl+C)")
    print("If model loading was interrupted, it will resume on next run.")
    sys.exit(130)


def download_models():
    """
    Pre-downloads all models required for Docling.
    Called before conversion to avoid hanging on first run.
    """
    from huggingface_hub import snapshot_download
    
    # All models used by Docling (including those loaded during convert)
    models_to_download = [
        # Main Docling models
        "ds4sd/docling-models",
        "docling-project/docling-models",
        # Layout models (used for structure recognition)
        "docling-project/docling-layout-heron",
        "ds4sd/docling-ibm-granite-dense-layout-heron",
        # TableFormer models (for table recognition)
        "ds4sd/docling-tableformer",
        "docling-project/tableformer",
        # RT-DETR models (element detection)
        "PekingU/rtdetr_r50vd",
    ]
    
    print("⏳ Checking and downloading models...")
    print("   (on first run, download may take a few minutes)\n")
    
    downloaded = 0
    for model_id in models_to_download:
        try:
            print(f"   📦 {model_id}...", end=" ", flush=True)
            snapshot_download(
                repo_id=model_id,
                local_files_only=False,
                resume_download=True,
            )
            print("✓")
            downloaded += 1
        except Exception as e:
            # Model may not exist or be unavailable
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str or "doesn't have" in error_str:
                print("skipped")
            else:
                print(f"⚠ error")
    
    print(f"\n✓ Models downloaded: {downloaded}\n")


def setup_device():
    """
    Device setup for processing.
    Automatically uses GPU on Apple M-series chips if available.
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Using Apple Silicon GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print(f"✓ Using CPU")
    
    return device


def convert_pdf_to_markdown(pdf_path: str, output_path: str = None) -> str:
    """
    Converts PDF file to Markdown format with full recognition.
    Always enabled: OCR for text and table structure recognition.
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output MD file (optional)
    
    Returns:
        Path to created MD file
    """
    # Check input file existence
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"File must have .pdf extension: {pdf_path}")
    
    # Determine output file
    if output_path is None:
        output_path = pdf_file.with_suffix('.md')
    else:
        output_path = Path(output_path)
    
    print(f"\n{'='*60}")
    print(f"Input file: {pdf_file.absolute()}")
    print(f"Output file: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Device setup for displaying information to the user
    # Note: Docling automatically uses available device through PyTorch
    device = setup_device()
    
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Pre-download all models
    download_models()
    
    # Configure PDF processing options
    # Enable OCR for scanned documents support
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for Russian text
    pipeline_options.do_table_structure = True  # Table structure recognition
    
    # Create converter with settings
    # Converter automatically uses best available device (MPS/CUDA/CPU)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    print("⏳ Starting PDF processing...")
    print("   This may take some time depending on document size...\n")
    
    # PDF conversion
    try:
        conversion_start = time.time()
        
        # Add callback for progress tracking
        print(f"[{time.strftime('%H:%M:%S')}] Starting conversion...")
        
        result = converter.convert(str(pdf_file.absolute()))
        
        conversion_time = time.time() - conversion_start
        print(f"[{time.strftime('%H:%M:%S')}] Conversion completed")
        
        print(f"\n✓ PDF processed in {conversion_time:.1f} seconds")
        
        # Export to Markdown
        print("⏳ Exporting to Markdown...")
        export_start = time.time()
        markdown_content = result.document.export_to_markdown()
        export_time = time.time() - export_start
        
        # Save to file
        output_path.write_text(markdown_content, encoding='utf-8')
        
        total_time = time.time() - conversion_start
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Result saved to: {output_path.absolute()}")
        print(f"\nStatistics:")
        print(f"  - Output file size: {output_path.stat().st_size / 1024:.2f} KB")
        print(f"  - Character count: {len(markdown_content)}")
        print(f"  - Conversion time: {conversion_time:.1f} sec")
        print(f"  - Export time: {export_time:.1f} sec")
        print(f"  - Total time: {total_time:.1f} sec")
        print(f"{'='*60}\n")
        
        return str(output_path.absolute())
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Processing interrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        print(f"\n✗ Conversion error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Convert PDF files to Markdown using Docling (IBM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s document.pdf
  %(prog)s document.pdf -o output.md
  %(prog)s /path/to/document.pdf -o /path/to/output.md

Supported features:
  - Complex document structures
  - Tables, images, formulas
  - OCR for scanned documents
  - Russian language
  - Acceleration on Apple M-series chips (GPU/NPU)
        """
    )
    
    parser.add_argument(
        'input_pdf',
        type=str,
        help='Path to input PDF file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output MD file (default: input file name with .md extension)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        convert_pdf_to_markdown(args.input_pdf, args.output)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
