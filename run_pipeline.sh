#!/bin/bash
# ==========================================
# AUTOMATED nnU-Net TRAINING PIPELINE
# ==========================================

# 1. SETUP LINGKUNGAN
echo "[1/5] Mengatur Environment Variables..."
export nnUNet_raw="/home/D13K48009/raid/nnUNet_raw"
export nnUNet_preprocessed="/home/D13K48009/raid/nnUNet_preprocessed"
export nnUNet_results="/home/D13K48009/raid/nnUNet_results"

echo "Mengaktifkan conda environment 'sinus_env'..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true
conda activate sinus_env

# 2. PLAN & PREPROCESS
echo "[2/5] Menjalankan nnUNetv2_plan_and_preprocess untuk Dataset 101, 102, 103..."
nnUNetv2_plan_and_preprocess -d 101 102 103 --verify_dataset_integrity

# 3. MENGAPLIKASIKAN STRATIFIED SPLITS
echo "[3/5] Menerapkan splits_final.json (Stratified 5-Fold) ke folder preprocessed..."
# Buat direktori jika belum ada secara paksa (aman)
mkdir -p $nnUNet_preprocessed/Dataset101_SinusExp1
mkdir -p $nnUNet_preprocessed/Dataset102_SinusExp2
mkdir -p $nnUNet_preprocessed/Dataset103_SinusExp3

cp $nnUNet_raw/Dataset101_SinusExp1/splits_final.json $nnUNet_preprocessed/Dataset101_SinusExp1/
cp $nnUNet_raw/Dataset102_SinusExp2/splits_final.json $nnUNet_preprocessed/Dataset102_SinusExp2/
cp $nnUNet_raw/Dataset103_SinusExp3/splits_final.json $nnUNet_preprocessed/Dataset103_SinusExp3/

# 4. TRAINING
# Secara default, kita jalankan FOLD 0 untuk Dataset 101 (Exp 1) demi mengejar preliminary results
FOLD=0
DATASET=101

echo "[4/5] Memulai Pelatihan nnU-Net (Dataset: ${DATASET}, Fold: ${FOLD})..."
echo "Mungkin akan memakan waktu yang sangat lama. Duduk manis!"
nnUNetv2_train ${DATASET} 3d_fullres ${FOLD} -tr nnUNetTrainer_250epochs

# 5. PREDICT PADA HELD-OUT TEST SET (30%)
echo "[5/5] Melakukan Prediksi pada Held-out Test Set & Evaluasi Metrik..."
# Pastikan folder output prediksi ada
OUTPUT_PRED="${nnUNet_results}/Dataset101_SinusExp1/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_${FOLD}/test_predictions"
mkdir -p $OUTPUT_PRED

# Folder images test set berada di folder utama (karena nnU-Net raw isinya hanya train set yang belum dipisah secara folder)
# Wait, nnU-Net butuh format _0000.nii.gz untuk predict.
# Kita akan buat folder temporary berisi 30% images test set
TEST_IMAGES_DIR="/home/D13K48009/raid/normal case_86/test_set_images"
mkdir -p $TEST_IMAGES_DIR

# Python untuk mengcopy test images
python3 -c "
import json
import os
import shutil
test_json = '/home/D13K48009/raid/normal case_86/held_out_test_set_30percent.json'
with open(test_json, 'r') as f:
    cases = json.load(f)['test_set']
src_dir = '/home/D13K48009/raid/normal case_86/imagesTr'
dst_dir = '${TEST_IMAGES_DIR}'
for c in cases:
    src = os.path.join(src_dir, f'{c}.nii.gz')
    dst = os.path.join(dst_dir, f'{c}_0000.nii.gz')
    if os.path.exists(src):
        shutil.copy2(src, dst)
"

# Jalankan Predict
nnUNetv2_predict -i $TEST_IMAGES_DIR -o $OUTPUT_PRED -d ${DATASET} -c 3d_fullres -f ${FOLD} -tr nnUNetTrainer_250epochs

# Bersihkan temporary test images
rm -rf $TEST_IMAGES_DIR

# Panggil script plotting
echo "Menggambar metrik akhir..."
python3 evaluate_metrics.py --dataset ${DATASET} --fold ${FOLD}

echo "=========================================="
echo "PIPELINE SELESAI!"
echo "Plot tersimpan di folder results_plots/"
echo "=========================================="
