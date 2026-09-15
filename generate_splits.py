import os
import json
import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

def main():
    base_dir = os.path.expanduser("~/raid/normal case_86")
    labels_exp3_dir = os.path.join(base_dir, "labels_exp3")
    
    # Ambil list pasien yang valid dari folder labels_exp3 (karena yg tidak valid sudah tidak ada di sana)
    valid_files = glob.glob(os.path.join(labels_exp3_dir, "*.nii.gz"))
    patients = []
    has_fsc = []
    
    print("Mengekstrak label FSC untuk stratifikasi...")
    for filepath in tqdm(valid_files):
        patient_id = os.path.basename(filepath).split("-label")[0]
        
        # Buka NIfTI dan cek apakah ada label 5 (FSC)
        img = nib.load(filepath)
        data = np.asarray(img.dataobj)  # Lebih hemat RAM dari get_fdata()
        
        patients.append(patient_id)
        # Cek jika ada voxel dengan nilai 5 (FSC)
        if 5 in data:
            has_fsc.append(1)
        else:
            has_fsc.append(0)
            
    patients = np.array(patients)
    has_fsc = np.array(has_fsc)
    
    print(f"\nTotal Pasien Valid: {len(patients)}")
    print(f"Jumlah pasien DENGAN FSC: {sum(has_fsc)}")
    print(f"Jumlah pasien TANPA FSC: {len(has_fsc) - sum(has_fsc)}")
    
    # 1. SPLIT 70% TRAIN, 30% TEST (SESUAI INSTRUKSI PPTX SLIDE 6)
    # Kita stratifikasi agar distribusi FSC seimbang di train dan test
    train_patients, test_patients, train_fsc, _ = train_test_split(
        patients, has_fsc, test_size=0.30, random_state=42, stratify=has_fsc
    )
    
    print(f"\nDistribusi Split:")
    print(f"Data Training & Validasi (70%): {len(train_patients)} kasus")
    print(f"Data Testing Held-Out (30%): {len(test_patients)} kasus")
    
    # 2. 5-FOLD CROSS VALIDATION PADA 70% DATA TRAIN (SESUAI METHODS_DRAFT)
    # Stratifikasi agar FSC terdistribusi merata di 5 fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    splits_final = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_patients, train_fsc)):
        train_fold = train_patients[train_idx].tolist()
        val_fold = train_patients[val_idx].tolist()
        splits_final.append({
            "train": train_fold,
            "val": val_fold
        })
        
    # 3. SIMPAN HASIL SPLIT UNTUK nnU-Net
    # nnU-Net mengharapkan file splits_final.json ditaruh di dalam folder dataset saat pra-pemrosesan.
    # Kita akan simpan di folder dasar dulu agar mudah diakses.
    
    nnunet_raw_dir = os.path.expanduser("~/raid/nnUNet_raw")
    
    # Simpan Splits untuk Training nnU-Net ke masing-masing folder Dataset
    # (Hanya data train/val 70% yang masuk ke splits_final)
    datasets = ["Dataset101_SinusExp1", "Dataset102_SinusExp2", "Dataset103_SinusExp3"]
    
    for ds in datasets:
        ds_dir = os.path.join(nnunet_raw_dir, ds)
        if os.path.exists(ds_dir):
            with open(os.path.join(ds_dir, "splits_final.json"), "w") as f:
                json.dump(splits_final, f, indent=4)
                
    # Simpan catatan held-out test set
    with open(os.path.join(base_dir, "held_out_test_set_30percent.json"), "w") as f:
        json.dump({"test_set": test_patients.tolist()}, f, indent=4)

    print("\nFile splits_final.json (5-fold) dan test set berhasil dibuat!")
    print("Silakan salin splits_final.json ke folder nnUNet_preprocessed/DatasetXXX setelah menjalankan nnUNetv2_plan_and_preprocess")

if __name__ == "__main__":
    main()
