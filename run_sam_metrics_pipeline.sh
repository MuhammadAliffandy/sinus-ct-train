#!/bin/bash
# ==========================================
# SAM-Med3D STANDALONE EVALUATION PIPELINE
# ==========================================
# Tidak membutuhkan: medim, torchio, pandas
# Hanya butuh: torch, nibabel, numpy, scipy, tqdm

echo "============================================="
echo "  SAM-Med3D Standalone Evaluation Pipeline"
echo "============================================="

echo ""
echo "Step 0: Memperbaiki numpy yang rusak..."
pip install numpy==1.26.4 --force-reinstall --quiet

echo ""
echo "Step 1: Memastikan scipy terinstal (untuk HD95 & ASD)..."
pip install scipy --quiet

echo ""
echo "Step 2: Memisahkan data validasi..."
cd /home/D13K48009/raid/Clara/sinus-ct-train
python split_sam_validation.py

echo ""
echo "Step 3: Menjalankan Inference + Kalkulasi Metrik (ALL-IN-ONE)..."
python evaluate_sam_standalone.py

echo ""
echo "============================================="
echo "Pipeline selesai!"
echo "Hasil tabel: /home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results/sam_metrics_table.csv"
echo "============================================="
