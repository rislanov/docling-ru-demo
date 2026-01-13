#!/usr/bin/env python3
"""
Скрипт для тестирования и отладки распознавания PDF файлов на русском языке
с помощью Docling от IBM.

Поддерживает:
- Сложные структуры документов
- Таблицы, изображения, формулы
- Apple M-series чипы (GPU через MPS backend)
- Русский язык (встроенная поддержка OCR)
"""

import argparse
import sys
from pathlib import Path
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def setup_device():
    """
    Настройка устройства для обработки.
    Автоматически использует GPU на Apple M-series чипах если доступно.
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Используется Apple Silicon GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Используется NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print(f"✓ Используется CPU")
    
    return device


def convert_pdf_to_markdown(pdf_path: str, output_path: str = None) -> str:
    """
    Конвертирует PDF файл в Markdown формат.
    
    Args:
        pdf_path: Путь к входному PDF файлу
        output_path: Путь к выходному MD файлу (опционально)
    
    Returns:
        Путь к созданному MD файлу
    """
    # Проверка существования входного файла
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"Файл должен иметь расширение .pdf: {pdf_path}")
    
    # Определение выходного файла
    if output_path is None:
        output_path = pdf_file.with_suffix('.md')
    else:
        output_path = Path(output_path)
    
    print(f"\n{'='*60}")
    print(f"Входной файл: {pdf_file.absolute()}")
    print(f"Выходной файл: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Настройка устройства для отображения информации пользователю
    # Примечание: Docling автоматически использует доступное устройство через PyTorch
    device = setup_device()
    
    # Настройка опций для обработки PDF
    # Включаем OCR для поддержки отсканированных документов
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Включить OCR для русского текста
    pipeline_options.do_table_structure = True  # Распознавание структуры таблиц
    
    # Создание конвертера с настройками
    # Конвертер автоматически использует лучшее доступное устройство (MPS/CUDA/CPU)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    print("Начинаем обработку PDF...")
    print("Это может занять некоторое время в зависимости от размера и сложности документа...\n")
    
    # Конвертация PDF
    try:
        result = converter.convert(str(pdf_file.absolute()))
        
        # Экспорт в Markdown
        markdown_content = result.document.export_to_markdown()
        
        # Сохранение в файл
        output_path.write_text(markdown_content, encoding='utf-8')
        
        print(f"✓ Успешно конвертировано!")
        print(f"✓ Результат сохранён в: {output_path.absolute()}")
        
        # Статистика
        print(f"\nСтатистика:")
        print(f"  - Размер выходного файла: {output_path.stat().st_size / 1024:.2f} КБ")
        print(f"  - Количество символов: {len(markdown_content)}")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"✗ Ошибка при конвертации: {e}", file=sys.stderr)
        raise


def main():
    """Основная функция с парсингом аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Конвертация PDF файлов в Markdown с помощью Docling (IBM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s document.pdf
  %(prog)s document.pdf -o output.md
  %(prog)s /path/to/document.pdf -o /path/to/output.md

Поддерживаемые возможности:
  - Сложные структуры документов
  - Таблицы, изображения, формулы
  - OCR для отсканированных документов
  - Русский язык
  - Ускорение на Apple M-series чипах (GPU/NPU)
        """
    )
    
    parser.add_argument(
        'input_pdf',
        type=str,
        help='Путь к входному PDF файлу'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Путь к выходному MD файлу (по умолчанию: имя входного файла с расширением .md)'
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
        print(f"\n✗ Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
