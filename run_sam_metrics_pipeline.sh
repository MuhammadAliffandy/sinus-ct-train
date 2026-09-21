#!/bin/bash
# ==========================================
# SAM-Med3D NATIVE EVALUATION PIPELINE
# ==========================================

echo "============================================="
echo "  Menggunakan Pipeline Asli SAM-Med3D"
echo "============================================="

echo "Step 0: Memperbaiki dependensi torchio (downgrade ke versi kompatibel SAM)..."
pip install torchio==0.18.90 --quiet

echo ""
echo "Step 1: Menjalankan Inference menggunakan Native SAM-Med3D..."
cd /home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D/
# Gunakan infer_sam_med3d.py asli (bukan standalone) agar resize dan spacing-nya akurat
cp /home/D13K48009/raid/Clara/sinus-ct-train/infer_sam_med3d.py .
python infer_sam_med3d.py

echo ""
echo "Step 2: Menghitung Metrik (Dice, IoU, Precision, ASD, HD95)..."
cd /home/D13K48009/raid/Clara/sinus-ct-train
python compute_sam_metrics.py

echo ""
echo "============================================="
echo "Pipeline selesai!"
echo "Hasil tabel: /home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results/sam_metrics_table.csv"
echo "============================================="
