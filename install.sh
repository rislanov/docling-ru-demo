#!/bin/bash
# Example script for installation and basic usage

echo "======================================"
echo "Docling RU Demo - Installation Example"
echo "======================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version

echo ""
echo "2. Installing dependencies..."
echo "   (This may take a few minutes)"
if ! pip3 install -r requirements.txt; then
    echo ""
    echo "✗ Error installing dependencies!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "3. Checking installed dependencies..."
if python3 check_deps.py; then
    echo ""
    echo "======================================"
    echo "Installation completed successfully!"
    echo "======================================"
    echo ""
    echo "Now you can use the script:"
    echo "  python3 pdf_to_md.py <your-file>.pdf"
    echo ""
    echo "For help:"
    echo "  python3 pdf_to_md.py --help"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ Installation not completed!"
    echo "======================================"
    echo ""
    echo "Some dependencies were not installed correctly."
    echo "Try installing them manually:"
    echo "  pip3 install -r requirements.txt"
    echo ""
    echo "Or update pip3 and try again:"
    echo "  pip3 install --upgrade pip"
    echo "  pip3 install -r requirements.txt"
    echo ""
    exit 1
fi
