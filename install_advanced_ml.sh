#!/bin/bash

echo "=========================================="
echo "Installing Advanced ML Dependencies"
echo "=========================================="
echo ""

echo "1. Installing CatBoost..."
pip install catboost>=1.2

echo ""
echo "2. Installing LightGBM..."
pip install lightgbm>=4.0

echo ""
echo "3. Verifying installations..."
python3 -c "import catboost; print(f'✓ CatBoost version: {catboost.__version__}')"
python3 -c "import lightgbm; print(f'✓ LightGBM version: {lightgbm.__version__}')"

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  - notebooks/12a_catboost_model.ipynb"
echo "  - notebooks/12b_lightgbm_model.ipynb"
echo "  - notebooks/13_time_series_cross_validation.ipynb"
