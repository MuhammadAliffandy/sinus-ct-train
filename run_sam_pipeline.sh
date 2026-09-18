#!/bin/bash
# ==========================================
# SAM-Med3D FINE-TUNING PIPELINE
# ==========================================

SAM_DIR="/home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D"
RAW_DATA_DIR="/home/D13K48009/raid/nnUNet_raw/Dataset101_SinusExp1"
SAM_DATA_ANT="/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior"
SAM_DATA_POST="/home/D13K48009/raid/SAM_Dataset/Exp1_Posterior"
WEIGHTS_PATH="${SAM_DIR}/sam_med3d_turbo.pth"

echo "1. Menyiapkan folder imagesTr untuk SAM-Med3D (menghapus akhiran _0000 dari format nnU-Net)..."
# Hapus symlink lama jika ada
rm -rf "$SAM_DATA_ANT/imagesTr" "$SAM_DATA_POST/imagesTr"
mkdir -p "$SAM_DATA_ANT/imagesTr"
mkdir -p "$SAM_DATA_POST/imagesTr"

# Copy dan hilangkan _0000 agar namanya sama persis dengan label
for img in "$RAW_DATA_DIR/imagesTr"/*_0000.nii.gz; do
    filename=$(basename "$img")
    new_filename=${filename/_0000/}
    if [ ! -f "$SAM_DATA_ANT/imagesTr/$new_filename" ]; then
        cp "$img" "$SAM_DATA_ANT/imagesTr/$new_filename"
    fi
    if [ ! -f "$SAM_DATA_POST/imagesTr/$new_filename" ]; then
        cp "$img" "$SAM_DATA_POST/imagesTr/$new_filename"
    fi
done

echo "2. Menyisipkan path dataset ke dalam kode sumber SAM-Med3D (utils/data_paths.py)..."
# SAM-Med3D tidak pakai argumen --data_dir, melainkan membaca langsung dari file data_paths.py
cat <<EOF > ${SAM_DIR}/utils/data_paths.py
img_datas = [
    '${SAM_DATA_ANT}',
]
EOF

echo "3. Mem-patch train.py agar tidak crash pada Dice Loss (mengonversi tensor label menjadi float)..."
cd ${SAM_DIR}
sed -i 's/loss = self.seg_loss(prev_masks, gt3D)/loss = self.seg_loss(prev_masks, gt3D.float())/g' train.py

echo "4. Memulai Fine-Tuning SAM-Med3D (Kelas Anterior)..."
# Kita jalankan 100 epoch dengan batch size 1. 
python train.py \
    --task_name "Sinus_Anterior" \
    --checkpoint "${WEIGHTS_PATH}" \
    --work_dir "/home/D13K48009/raid/SAM_results/Exp1_Anterior" \
    --num_epochs 100 \
    --batch_size 1 \
    --lr 0.0001 \
    --num_workers 4

echo "Selesai! Model tersimpan di folder SAM_results."
