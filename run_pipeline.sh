#!/bin/bash
# ==========================================
# AUTOMATED nnU-Net TRAINING PIPELINE
# ==========================================

# Default values
DATASET=101
FOLD=0
TRAINER="nnUNetTrainer" # Default ke 1000 epoch
CONTINUE_FLAG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dataset) DATASET="$2"; shift ;;
        -f|--fold) FOLD="$2"; shift ;;
        -tr|--trainer) TRAINER="$2"; shift ;;
        -c|--continue) CONTINUE_FLAG="--c" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Menjalankan Pipeline dengan parameter:"
echo "Dataset: ${DATASET}"
echo "Fold: ${FOLD}"
echo "Trainer: ${TRAINER}"
echo "=========================================="

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
# Selalu pastikan integrity dataset
nnUNetv2_plan_and_preprocess -d 101 102 103 --verify_dataset_integrity

# 3. MENGAPLIKASIKAN STRATIFIED SPLITS
echo "[3/5] Menerapkan splits_final.json (Stratified 5-Fold) ke folder preprocessed..."
mkdir -p "$nnUNet_preprocessed/Dataset101_SinusExp1"
mkdir -p "$nnUNet_preprocessed/Dataset102_SinusExp2"
mkdir -p "$nnUNet_preprocessed/Dataset103_SinusExp3"

cp "$nnUNet_raw/Dataset101_SinusExp1/splits_final.json" "$nnUNet_preprocessed/Dataset101_SinusExp1/"
cp "$nnUNet_raw/Dataset102_SinusExp2/splits_final.json" "$nnUNet_preprocessed/Dataset102_SinusExp2/"
cp "$nnUNet_raw/Dataset103_SinusExp3/splits_final.json" "$nnUNet_preprocessed/Dataset103_SinusExp3/"

# Set nama eksperimen
if [ "$DATASET" == "101" ]; then
    EXP_NAME="Dataset101_SinusExp1"
elif [ "$DATASET" == "102" ]; then
    EXP_NAME="Dataset102_SinusExp2"
elif [ "$DATASET" == "103" ]; then
    EXP_NAME="Dataset103_SinusExp3"
else
    echo "Dataset tidak valid."
    exit 1
fi

# 4. TRAINING
echo "[4/5] Memulai Pelatihan nnU-Net (Dataset: ${DATASET}, Fold: ${FOLD}, Trainer: ${TRAINER})..."
echo "Mungkin akan memakan waktu yang sangat lama. Duduk manis!"

if [ "$FOLD" == "all" ]; then
    for i in {0..4}; do
        echo "--> Training Fold $i..."
        nnUNetv2_train ${DATASET} 3d_fullres ${i} -tr ${TRAINER} ${CONTINUE_FLAG}
    done
else
    nnUNetv2_train ${DATASET} 3d_fullres ${FOLD} -tr ${TRAINER} ${CONTINUE_FLAG}
fi

# 5. PREDICT PADA HELD-OUT TEST SET (30%)
echo "[5/5] Melakukan Prediksi pada Held-out Test Set & Evaluasi Metrik..."
TEST_IMAGES_DIR="/home/D13K48009/raid/normal case_86/test_set_images"
mkdir -p "$TEST_IMAGES_DIR"

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

if [ "$FOLD" == "all" ]; then
    OUTPUT_PRED="${nnUNet_results}/${EXP_NAME}/${TRAINER}__nnUNetPlans__3d_fullres/ensemble_predictions"
    mkdir -p "$OUTPUT_PRED"
    echo "--> Melakukan Prediksi Secara Ensemble (Fold 0-4)..."
    nnUNetv2_predict -i "$TEST_IMAGES_DIR" -o "$OUTPUT_PRED" -d ${DATASET} -c 3d_fullres -f 0 1 2 3 4 -tr ${TRAINER}
else
    OUTPUT_PRED="${nnUNet_results}/${EXP_NAME}/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/test_predictions"
    mkdir -p "$OUTPUT_PRED"
    echo "--> Melakukan Prediksi Pada Fold ${FOLD}..."
    nnUNetv2_predict -i "$TEST_IMAGES_DIR" -o "$OUTPUT_PRED" -d ${DATASET} -c 3d_fullres -f ${FOLD} -tr ${TRAINER}
fi

# Bersihkan temporary test images
rm -rf "$TEST_IMAGES_DIR"

# Panggil script plotting
echo "Menggambar metrik akhir..."
python3 evaluate_metrics.py --dataset ${DATASET} --fold ${FOLD} --trainer ${TRAINER}

echo "=========================================="
echo "PIPELINE SELESAI!"
echo "Plot tersimpan di folder results_plots/"
echo "=========================================="
