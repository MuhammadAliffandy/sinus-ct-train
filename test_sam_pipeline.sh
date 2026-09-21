#!/bin/bash
# ==========================================
# SAM-Med3D INFERENCE & EVALUATION PIPELINE
# ==========================================

SAM_DIR="/home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D"
SAM_DATA_ANT="/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior"
CHECKPOINT="/home/D13K48009/raid/SAM_results/Exp1_Anterior/Sinus_Anterior/sam_model_dice_best.pth"
RESULTS_DIR="/home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results"

echo "1. Menyiapkan direktori hasil pengujian..."
mkdir -p "$RESULTS_DIR"

echo "2. Memulai proses Inference dan Kalkulasi Akurasi (Dice Score)..."
cd "$SAM_DIR"

# Menjalankan script validasi bawaan SAM-Med3D dengan 1 titik klik (prompt)
python validation.py \
    --seed 2026 \
    -vp "$RESULTS_DIR" \
    -cp "$CHECKPOINT" \
    -tdp "$SAM_DATA_ANT" \
    -nc 1 \
    --save_name "$RESULTS_DIR/test_metrics.txt"

echo "=========================================="
echo "Selesai! Hasil akurasi eksak telah disimpan di:"
echo "$RESULTS_DIR/test_metrics.txt"
echo "Silakan cek file tersebut (bisa gunakan perintah cat) untuk melihat skor Dice-nya."
