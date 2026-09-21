#!/bin/bash
# ==========================================
# SAM-Med3D INFERENCE & METRICS PIPELINE
# ==========================================

echo "1. Mempersiapkan folder dataset validasi..."
python split_sam_validation.py

echo "2. Memastikan pustaka 'medim' terinstal..."
# Dihapus agar tidak menimpa instalasi PyTorch bawaan DGX

echo "3. Menjalankan proses Inferensi SAM-Med3D (membuat prediksi 3D)..."
# Jalankan inferensi dari dalam folder SAM-Med3D agar bisa membaca utils
cp infer_sam_med3d.py /home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D/
cd /home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D/
python infer_sam_med3d.py

echo "4. Menghitung Metrik Lengkap (Dice, IoU, Precision, ASD, HD95)..."
# Kembali ke root folder
cd /home/D13K48009/raid/Clara/sinus-ct-train
python compute_sam_metrics.py

echo "=========================================="
echo "Selesai! Tabel CSV sudah berhasil di-generate."
echo "Buka file /home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results/sam_metrics_table.csv"
