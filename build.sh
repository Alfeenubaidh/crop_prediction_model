#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Fetching model artifacts..."
mkdir -p models

curl -fL "$MODELS_ZIP_URL" -o models.zip
unzip -o models.zip -d models/
rm models.zip

echo "All model artifacts ready."