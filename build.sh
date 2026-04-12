#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo "Fetching model artifacts..."
mkdir -p backend/models

curl -fL "$MODELS_ZIP_URL" -o models.zip
unzip -o models.zip -d backend/models/
rm models.zip

echo "All model artifacts ready."