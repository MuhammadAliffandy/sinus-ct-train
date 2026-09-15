#!/bin/bash
# Script untuk setup Conda Environment di Server DGX

echo "Membuat conda environment bernama 'sinus_env' dengan Python 3.10..."
# Pastikan Anda memiliki koneksi internet, tekan 'y' jika diminta konfirmasi
conda create -n sinus_env python=3.10 -y

echo "========================================================="
echo "Lingkungan conda berhasil dibuat!"
echo "Karena keterbatasan script bash, Anda harus mengaktifkannya sendiri dengan mengetik:"
echo "conda activate sinus_env"
echo ""
echo "Setelah masuk ke environment (akan ada tulisan (sinus_env) di kiri terminal), jalankan:"
echo "pip install -r requirements.txt"
echo "python prepare_labels.py"
echo "========================================================="
