import os
import json
import shutil
from pathlib import Path

def split_sam_validation():
    # Paths (disesuaikan dengan DGX environment)
    splits_file = "/home/D13K48009/raid/nnUNet_preprocessed/Dataset101_SinusExp1/splits_final.json"
    sam_data_ant = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior"
    sam_data_post = "/home/D13K48009/raid/SAM_Dataset/Exp1_Posterior"
    
    # Baca file splits_final.json (kita ambil fold 0)
    print(f"Reading splits from {splits_file}...")
    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        val_cases = splits[0]['val']
        print(f"Found {len(val_cases)} validation cases for Fold 0.")
    except Exception as e:
        print(f"Error reading splits file: {e}")
        return

    # Helper function to move validation cases
    def move_to_val(base_path, cases):
        base_path = Path(base_path)
        img_tr = base_path / "imagesTr"
        lbl_tr = base_path / "labelsTr"
        img_val = base_path / "imagesVal"
        lbl_val = base_path / "labelsVal"
        
        img_val.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for case in cases:
            # Pindahkan image (hilangkan _0000 karena sudah di-rename sebelumnya)
            src_img = img_tr / f"{case}.nii.gz"
            dst_img = img_val / f"{case}.nii.gz"
            
            # Pindahkan label
            src_lbl = lbl_tr / f"{case}.nii.gz"
            dst_lbl = lbl_val / f"{case}.nii.gz"
            
            if src_img.exists() and src_lbl.exists():
                shutil.move(str(src_img), str(dst_img))
                shutil.move(str(src_lbl), str(dst_lbl))
                count += 1
            else:
                # Jika sudah di folder val, lewati
                if dst_img.exists():
                    count += 1
                else:
                    print(f"Warning: {case} tidak ditemukan di {base_path}")
        print(f"Berhasil menyiapkan {count} kasus validasi di {base_path}")

    print("\nProcessing Anterior Class...")
    move_to_val(sam_data_ant, val_cases)
    
    print("\nProcessing Posterior Class...")
    move_to_val(sam_data_post, val_cases)
    
    print("\nPemisahan data validasi selesai!")

if __name__ == "__main__":
    split_sam_validation()
