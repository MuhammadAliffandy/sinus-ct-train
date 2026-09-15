#!/bin/bash
# Script untuk setup Virtual Environment di Server DGX

echo "Membuat virtual environment 'sinus_env'..."
python3 -m venv sinus_env

echo "Mengaktifkan virtual environment..."
source sinus_env/bin/activate

echo "Memperbarui pip..."
pip install --upgrade pip

echo "Menginstal requirements..."
pip install -r requirements.txt

echo "========================================================="
echo "Setup Selesai!"
echo "Untuk menjalankan script nanti, pastikan Anda masuk ke venv dengan perintah:"
echo "source sinus_env/bin/activate"
echo "Lalu jalankan:"
echo "python prepare_labels.py"
echo "========================================================="
