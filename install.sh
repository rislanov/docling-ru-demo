#!/bin/bash
# Пример скрипта для установки и базового использования

echo "======================================"
echo "Docling RU Demo - Пример установки"
echo "======================================"
echo ""

# Проверка Python версии
echo "1. Проверка версии Python..."
python3 --version

echo ""
echo "2. Установка зависимостей..."
echo "   (Это может занять несколько минут)"
if ! pip3 install -r requirements.txt; then
    echo ""
    echo "✗ Ошибка при установке зависимостей!"
    echo "Пожалуйста, проверьте сообщения об ошибках выше."
    exit 1
fi

echo ""
echo "3. Проверка установленных зависимостей..."
if python3 check_deps.py; then
    echo ""
    echo "======================================"
    echo "Установка завершена успешно!"
    echo "======================================"
    echo ""
    echo "Теперь вы можете использовать скрипт:"
    echo "  python3 pdf_to_md.py <ваш-файл>.pdf"
    echo ""
    echo "Для получения справки:"
    echo "  python3 pdf_to_md.py --help"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ Установка не завершена!"
    echo "======================================"
    echo ""
    echo "Некоторые зависимости не были установлены корректно."
    echo "Попробуйте установить их вручную:"
    echo "  pip3 install -r requirements.txt"
    echo ""
    echo "Или обновите pip3 и попробуйте снова:"
    echo "  pip3 install --upgrade pip"
    echo "  pip3 install -r requirements.txt"
    echo ""
    exit 1
fi
