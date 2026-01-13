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
import signal
import logging
import os
import time
from pathlib import Path

# Включаем отображение прогресса загрузки моделей HuggingFace
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # Стабильная загрузка
os.environ.setdefault("TQDM_DISABLE", "0")  # Включить прогресс-бары

import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# Настройка логирования для отображения прогресса загрузки моделей
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Увеличиваем детализацию логов для отладки проблем инициализации
logger = logging.getLogger()


def set_verbose_logging():
    """Включает детальное логирование для отладки."""
    logger.setLevel(logging.INFO)
    # Добавляем логирование для важных модулей
    logging.getLogger('docling').setLevel(logging.INFO)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)  # Уменьшаем шум от HF
    

set_verbose_logging()


def signal_handler(sig, frame):
    """Обработчик прерывания Ctrl+C."""
    print("\n\n⚠ Прервано пользователем (Ctrl+C)")
    print("Если загрузка моделей была прервана, при следующем запуске она продолжится.")
    sys.exit(130)


def download_models():
    """
    Предварительно загружает все модели, необходимые для Docling.
    Вызывается перед конвертацией, чтобы избежать зависания при первом запуске.
    """
    from huggingface_hub import snapshot_download
    
    # Все модели, используемые Docling (включая те, что загружаются при convert)
    models_to_download = [
        # Основные модели Docling
        "ds4sd/docling-models",
        "docling-project/docling-models",
        # Layout модели (используются для распознавания структуры)
        "docling-project/docling-layout-heron",
        "ds4sd/docling-ibm-granite-dense-layout-heron",
        # TableFormer модели (для распознавания таблиц)
        "ds4sd/docling-tableformer",
        "docling-project/tableformer",
        # RT-DETR модели (детекция элементов)
        "PekingU/rtdetr_r50vd",
    ]
    
    print("⏳ Проверка и загрузка моделей...")
    print("   (при первом запуске загрузка может занять несколько минут)\n")
    
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
            # Модель может не существовать или быть недоступной
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str or "doesn't have" in error_str:
                print("пропущено")
            else:
                print(f"⚠ ошибка")
    
    print(f"\n✓ Загружено моделей: {downloaded}\n")


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
    Конвертирует PDF файл в Markdown формат с полным распознаванием.
    Всегда включены: OCR для текста и распознавание структуры таблиц.
    
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
    
    # Регистрируем обработчик Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Предварительно загружаем все модели
    download_models()
    
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
    
    print("⏳ Начинаем обработку PDF...")
    print("   Это может занять некоторое время в зависимости от размера документа...\n")
    
    # Конвертация PDF
    try:
        conversion_start = time.time()
        
        # Добавляем callback для отслеживания прогресса
        print(f"[{time.strftime('%H:%M:%S')}] Начинаем конвертацию...")
        
        result = converter.convert(str(pdf_file.absolute()))
        
        conversion_time = time.time() - conversion_start
        print(f"[{time.strftime('%H:%M:%S')}] Конвертация завершена")
        
        print(f"\n✓ PDF обработан за {conversion_time:.1f} секунд")
        
        # Экспорт в Markdown
        print("⏳ Экспорт в Markdown...")
        export_start = time.time()
        markdown_content = result.document.export_to_markdown()
        export_time = time.time() - export_start
        
        # Сохранение в файл
        output_path.write_text(markdown_content, encoding='utf-8')
        
        total_time = time.time() - conversion_start
        
        print(f"\n{'='*60}")
        print(f"✓ УСПЕШНО ЗАВЕРШЕНО!")
        print(f"{'='*60}")
        print(f"Результат сохранён в: {output_path.absolute()}")
        print(f"\nСтатистика:")
        print(f"  - Размер выходного файла: {output_path.stat().st_size / 1024:.2f} КБ")
        print(f"  - Количество символов: {len(markdown_content)}")
        print(f"  - Время конвертации: {conversion_time:.1f} сек")
        print(f"  - Время экспорта: {export_time:.1f} сек")
        print(f"  - Общее время: {total_time:.1f} сек")
        print(f"{'='*60}\n")
        
        return str(output_path.absolute())
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Обработка прервана пользователем (Ctrl+C)")
        raise
    except Exception as e:
        print(f"\n✗ Ошибка при конвертации: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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
