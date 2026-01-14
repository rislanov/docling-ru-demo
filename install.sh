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
echo "2. Creating virtual environment..."
if ! python3 -m venv venv; then
    echo ""
    echo "✗ Error creating virtual environment!"
    echo "Please make sure python3-venv is installed."
    exit 1
fi

echo "   ✓ Virtual environment created in 'venv' directory"

echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate

echo "   ✓ Virtual environment activated"

echo ""
echo "4. Installing dependencies..."
echo "   (This may take a few minutes)"
if ! pip install -r requirements.txt; then
    echo ""
    echo "✗ Error installing dependencies!"
    echo "Please check the error messages above."
    deactivate
    exit 1
fi

echo ""
echo "5. Checking installed dependencies..."
if python check_deps.py; then
    echo ""
    echo "======================================"
    echo "Installation completed successfully!"
    echo "======================================"
    echo ""
    echo "To use the script, first activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Then run the script:"
    echo "  python pdf_to_md.py <your-file>.pdf"
    echo ""
    echo "For help:"
    echo "  python pdf_to_md.py --help"
    echo ""
    echo "To deactivate the virtual environment when done:"
    echo "  deactivate"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ Installation not completed!"
    echo "======================================"
    echo ""
    echo "Some dependencies were not installed correctly."
    echo "Try installing them manually:"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Or update pip and try again:"
    echo "  source venv/bin/activate"
    echo "  pip install --upgrade pip"
    echo "  pip install -r requirements.txt"
    echo ""
    deactivate
    exit 1
fi
