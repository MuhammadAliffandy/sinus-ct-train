import os
import glob
import json
import shutil
from tqdm import tqdm

def generate_dataset_json(target_dir, experiment_mode, num_training):
    if experiment_mode == 1:
        labels = {
            "background": 0,
            "Anterior": 1,
            "Posterior": 2
        }
    elif experiment_mode == 2:
        labels = {
            "background": 0,
            "Right Anterior": 1,
            "Left Anterior": 2,
            "Right Posterior": 3,
            "Left Posterior": 4
        }
    elif experiment_mode == 3:
        labels = {
            "background": 0,
            "Right Anterior": 1,
            "Left Anterior": 2,
            "Right Posterior": 3,
            "Left Posterior": 4,
            "FSC": 5
        }
        
    dataset_info = {
        "channel_names": {
            "0": "CT"
        },
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(target_dir, "dataset.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)

def main():
    base_dir = os.path.expanduser("~/raid/normal case_86")
    
    # Path Data Awal (sudah direlabel & lolos QC)
    images_dir = os.path.join(base_dir, "imagesTr")
    labels_exp1_dir = os.path.join(base_dir, "labels_exp1")
    labels_exp2_dir = os.path.join(base_dir, "labels_exp2")
    labels_exp3_dir = os.path.join(base_dir, "labels_exp3")
    
    # Path Output nnU-Net
    nnunet_raw_dir = os.path.expanduser("~/raid/nnUNet_raw")
    
    datasets = {
        1: os.path.join(nnunet_raw_dir, "Dataset101_SinusExp1"),
        2: os.path.join(nnunet_raw_dir, "Dataset102_SinusExp2"),
        3: os.path.join(nnunet_raw_dir, "Dataset103_SinusExp3")
    }
    
    # Buat folder-folder dasar nnU-Net
    for d in datasets.values():
        os.makedirs(os.path.join(d, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(d, "labelsTr"), exist_ok=True)
        
    # Daftar Pasien yang Gagal QC (Dihiraukan)
    mismatched_patients = ["6526305", "2388344"]
    
    # Ambil list pasien yang valid dari folder labels_exp1 (krn sdh di-QC pada script sebelumnya)
    valid_files = glob.glob(os.path.join(labels_exp1_dir, "*.nii.gz"))
    valid_patients = [os.path.basename(f).split("-label")[0] for f in valid_files]
    
    # Hapus yang mismatch dari list (jika masih ada)
    valid_patients = [p for p in valid_patients if p not in mismatched_patients]
    
    print(f"Total pasien valid (Lolos QC) untuk nnU-Net: {len(valid_patients)}")
    
    for exp_mode, dataset_dir in datasets.items():
        print(f"\nMenyiapkan {os.path.basename(dataset_dir)}...")
        
        # Bikin json
        generate_dataset_json(dataset_dir, exp_mode, len(valid_patients))
        
        # Tentukan folder sumber label
        if exp_mode == 1:
            src_labels_dir = labels_exp1_dir
        elif exp_mode == 2:
            src_labels_dir = labels_exp2_dir
        else:
            src_labels_dir = labels_exp3_dir
            
        for patient_id in tqdm(valid_patients):
            # Format nnU-Net wajib menggunakan akhiran _0000.nii.gz untuk image
            src_img = os.path.join(images_dir, f"{patient_id}.nii.gz")
            dst_img = os.path.join(dataset_dir, "imagesTr", f"{patient_id}_0000.nii.gz")
            
            src_lbl = os.path.join(src_labels_dir, f"{patient_id}-label.nii.gz")
            dst_lbl = os.path.join(dataset_dir, "labelsTr", f"{patient_id}.nii.gz") # Label tidak pakai 0000
            
            # Kita gunakan file symlink atau copy. Agar aman di semua file system, kita copy.
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    print("\nFormat dataset nnU-Net selesai dibuat!")

if __name__ == "__main__":
    main()
