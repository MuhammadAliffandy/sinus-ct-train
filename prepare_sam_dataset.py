import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse

def create_binary_masks(input_dir, output_dir_class1, output_dir_class2):
    """
    SAM-Med3D membutuhkan label berbentuk biner (0 dan 1).
    Fungsi ini memecah citra label NIfTI yang memiliki kelas 1 (Anterior) dan 2 (Posterior)
    menjadi dua file terpisah yang masing-masing murni biner.
    """
    os.makedirs(output_dir_class1, exist_ok=True)
    os.makedirs(output_dir_class2, exist_ok=True)

    label_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    if not label_files:
        print(f"Tidak ada file NIfTI yang ditemukan di {input_dir}")
        return

    print(f"Memproses {len(label_files)} file label menjadi format biner untuk SAM-Med3D...")
    
    for filepath in tqdm(label_files):
        filename = os.path.basename(filepath)
        
        # Baca citra menggunakan SimpleITK
        img = sitk.ReadImage(filepath)
        img_arr = sitk.GetArrayFromImage(img)
        
        # Buat mask biner untuk Kelas 1 (Anterior)
        mask1_arr = np.where(img_arr == 1, 1, 0).astype(np.uint8)
        mask1_img = sitk.GetImageFromArray(mask1_arr)
        mask1_img.CopyInformation(img)
        sitk.WriteImage(mask1_img, os.path.join(output_dir_class1, filename))
        
        # Buat mask biner untuk Kelas 2 (Posterior)
        mask2_arr = np.where(img_arr == 2, 1, 0).astype(np.uint8)
        mask2_img = sitk.GetImageFromArray(mask2_arr)
        mask2_img.CopyInformation(img)
        sitk.WriteImage(mask2_img, os.path.join(output_dir_class2, filename))

    print("✅ Konversi ke mask biner (Anterior dan Posterior) selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Multi-class Labels to Binary for SAM-Med3D")
    parser.add_argument("--input", type=str, required=True, help="Path ke folder label multi-class asli (contoh: labelsTr)")
    parser.add_argument("--out1", type=str, required=True, help="Path output untuk kelas Anterior (biner)")
    parser.add_argument("--out2", type=str, required=True, help="Path output untuk kelas Posterior (biner)")
    args = parser.parse_args()
    
    create_binary_masks(args.input, args.out1, args.out2)
