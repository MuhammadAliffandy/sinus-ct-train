import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

def get_patient_mapping(csv_path):
    """
    Membaca ColorTable CSV dari pasien dan mengembalikan kamus pemetaan (dictionary)
    dari string 'Name' ke integer 'LabelValue'.
    """
    try:
        # Coba load dengan delimiter koma terlebih dahulu
        df = pd.read_csv(csv_path)
        # Jika kolom 'Name' tidak terbaca, kemungkinan menggunakan pemisah spasi (tab/whitespace)
        if 'Name' not in df.columns:
            df = pd.read_csv(csv_path, sep=r'\s+')
            
        mapping = {}
        for _, row in df.iterrows():
            name = str(row['Name']).strip()
            val = int(row['LabelValue'])
            mapping[name] = val
        return mapping
    except Exception as e:
        print(f"Error saat membaca {csv_path}: {e}")
        return {}

def remap_patient_data(data, name_to_val, exp_mode):
    """
    Mengubah array nilai label (data) pasien menjadi kelas eksperimen
    berdasarkan string 'Name' struktur anatominya (Bukan LabelValue mentah).
    """
    # Pengelompokan Kategori Berdasarkan Nama Struktur
    categories = {
        'R-Ant': ['RANC', 'RSAC', 'RSAFC'],
        'L-Ant': ['LANC', 'LSAC', 'LSAFC'],
        'R-Post': ['RBE', 'RSBC', 'RSBFC', 'RSOEC'],
        'L-Post': ['LBE', 'LSBC', 'LSBFC', 'LSOEC'],
        'FSC': ['FSC']
    }
    
    # Tentukan output label untuk mode eksperimen yang dipilih
    target_labels = {}
    if exp_mode == 1:
        target_labels['R-Ant'] = 1
        target_labels['L-Ant'] = 1
        target_labels['R-Post'] = 2
        target_labels['L-Post'] = 2
        # FSC dibiarkan tidak ter-map agar otomatis menjadi 0 (background)
    elif exp_mode == 2:
        target_labels['R-Ant'] = 1
        target_labels['L-Ant'] = 2
        target_labels['R-Post'] = 3
        target_labels['L-Post'] = 4
        # FSC dibiarkan tidak ter-map
    elif exp_mode == 3:
        target_labels['R-Ant'] = 1
        target_labels['L-Ant'] = 2
        target_labels['R-Post'] = 3
        target_labels['L-Post'] = 4
        target_labels['FSC'] = 5
        
    remapped_data = np.zeros_like(data)
    
    # Proses konversi
    for cat_name, structures in categories.items():
        if cat_name in target_labels:
            target_val = target_labels[cat_name]
            for struct in structures:
                if struct in name_to_val:
                    orig_val = name_to_val[struct]
                    remapped_data[data == orig_val] = target_val
                    
    return remapped_data

def main():
    # PATH KE DATASET DI SERVER DGX
    base_dir = os.path.expanduser("~/raid/normal case_86")
    
    labels_dir = os.path.join(base_dir, "labelsTr")
    images_dir = os.path.join(base_dir, "imagesTr")
    color_tables_dir = os.path.join(base_dir, "ColorTable")
    
    output_dirs = {
        1: os.path.join(base_dir, "labels_exp1"),
        2: os.path.join(base_dir, "labels_exp2"),
        3: os.path.join(base_dir, "labels_exp3")
    }
    
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    label_files = glob.glob(os.path.join(labels_dir, "*-label.nii.gz"))
    print(f"Ditemukan {len(label_files)} file label di {labels_dir}")
    
    mismatch_log_path = os.path.join(base_dir, "mismatched_cases.txt")
    # Bersihkan file log lama jika ada
    if os.path.exists(mismatch_log_path):
        os.remove(mismatch_log_path)
    
    for filepath in tqdm(label_files, desc="Memproses Pasien"):
        filename = os.path.basename(filepath)
        # Ambil ID Pasien (misal dari "1192935-label.nii.gz" menjadi "1192935")
        patient_id = filename.split('-label')[0] 
        
        color_table_path = os.path.join(color_tables_dir, f"{patient_id}_ColorTable.csv")
        image_path = os.path.join(images_dir, f"{patient_id}.nii.gz")
        
        if not os.path.exists(color_table_path):
            print(f"\nWARNING: ColorTable untuk pasien {patient_id} tidak ditemukan. Dilewati.")
            continue
            
        if not os.path.exists(image_path):
            print(f"\nWARNING: File image ({patient_id}.nii.gz) tidak ditemukan. Dilewati.")
            continue
            
        name_to_val = get_patient_mapping(color_table_path)
        if not name_to_val:
            print(f"\nWARNING: Gagal membaca ColorTable untuk pasien {patient_id}. Dilewati.")
            continue
            
        # Buka NIfTI Label dan Image
        label_img = nib.load(filepath)
        raw_img = nib.load(image_path)
        
        # PENGECEKAN QC (QUALITY CONTROL) SESUAI INSTRUKSI KLIEN (SLIDE 6 & METHODS)
        # 1. Cek Dimensi (Size)
        if label_img.shape != raw_img.shape:
            msg = f"MISMATCH DIMENSI: Pasien {patient_id} (Label: {label_img.shape} vs Image: {raw_img.shape})"
            with open(mismatch_log_path, "a") as f:
                f.write(msg + "\n")
            continue
            
        # 2. Cek Spacing (Jarak Antar Piksel/Voxel)
        if not np.allclose(label_img.header.get_zooms(), raw_img.header.get_zooms(), atol=1e-3):
            msg = f"MISMATCH SPACING: Pasien {patient_id} (Label: {label_img.header.get_zooms()} vs Image: {raw_img.header.get_zooms()})"
            with open(mismatch_log_path, "a") as f:
                f.write(msg + "\n")
            continue
            
        data = label_img.get_fdata()
        
        # Jalankan ke 3 eksperimen
        for exp_mode in [1, 2, 3]:
            remapped_data = remap_patient_data(data, name_to_val, exp_mode)
            img_exp = nib.Nifti1Image(remapped_data.astype(np.uint8), label_img.affine, label_img.header)
            output_path = os.path.join(output_dirs[exp_mode], filename)
            nib.save(img_exp, output_path)

    print("\nProses Re-labeling Dinamis Berdasarkan ColorTable Pasien Selesai!")
    if os.path.exists(mismatch_log_path):
        print(f"Perhatian: Terdapat kasus dengan masalah dimensi/spacing. Silakan cek file: {mismatch_log_path}")

if __name__ == "__main__":
    main()
