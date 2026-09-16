import os
import glob
import json
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_dice(pred, gt, classes):
    """
    Menghitung skor Dice (DSC) per kelas.
    """
    dice_scores = {}
    for c in classes:
        if c == 0: continue # Skip background
        p_c = (pred == c)
        g_c = (gt == c)
        
        intersection = np.logical_and(p_c, g_c).sum()
        union = p_c.sum() + g_c.sum()
        
        if union == 0:
            # Jika struktur anatomi benar-benar tidak ada di ground truth maupun prediksi
            dice = np.nan 
        else:
            dice = 2.0 * intersection / union
        dice_scores[c] = dice
    return dice_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=101, help='ID Dataset (misal 101)')
    parser.add_argument('--fold', type=str, default="0", help='Fold yang dilatih (misal 0 atau all)')
    parser.add_argument('--trainer', type=str, default="nnUNetTrainer", help='Nama Trainer nnU-Net')
    args = parser.parse_args()
    
    # Path Setup
    base_dir = os.path.expanduser("~/raid/normal case_86")
    results_dir = os.path.expanduser("~/raid/nnUNet_results")
    
    if args.dataset == 101:
        gt_dir = os.path.join(base_dir, "labels_exp1")
        class_names = {1: "Anterior", 2: "Posterior"}
        exp_folder = "Dataset101_SinusExp1"
    elif args.dataset == 102:
        gt_dir = os.path.join(base_dir, "labels_exp2")
        class_names = {1: "R-Ant", 2: "L-Ant", 3: "R-Post", 4: "L-Post"}
        exp_folder = "Dataset102_SinusExp2"
    elif args.dataset == 103:
        gt_dir = os.path.join(base_dir, "labels_exp3")
        class_names = {1: "R-Ant", 2: "L-Ant", 3: "R-Post", 4: "L-Post", 5: "FSC"}
        exp_folder = "Dataset103_SinusExp3"
    else:
        raise ValueError("Dataset tidak dikenali.")

    if args.fold == "all":
        pred_dir = os.path.join(
            results_dir, 
            exp_folder, 
            f"{args.trainer}__nnUNetPlans__3d_fullres/ensemble_predictions"
        )
    else:
        pred_dir = os.path.join(
            results_dir, 
            exp_folder, 
            f"{args.trainer}__nnUNetPlans__3d_fullres/fold_{args.fold}/test_predictions"
        )
    
    out_plot_dir = os.path.join(base_dir, "..", "Clara", "sinus-ct-train", "results_plots")
    os.makedirs(out_plot_dir, exist_ok=True)
    
    print(f"Mengambil prediksi dari: {pred_dir}")
    pred_files = glob.glob(os.path.join(pred_dir, "*.nii.gz"))
    
    all_scores = []
    
    for pf in pred_files:
        patient_id = os.path.basename(pf).replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{patient_id}-label.nii.gz")
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth tidak ditemukan untuk test case {patient_id}")
            continue
            
        pred_img = nib.load(pf).get_fdata()
        gt_img = nib.load(gt_path).get_fdata()
        
        # Pengecekan dimensi untuk keamanan (Meski harusnya sama karena sudah QC)
        if pred_img.shape != gt_img.shape:
            continue
            
        dice_dict = compute_dice(pred_img, gt_img, class_names.keys())
        
        for c_id, c_name in class_names.items():
            all_scores.append({
                "Patient": patient_id,
                "Class": c_name,
                "Dice": dice_dict[c_id]
            })
            
    df = pd.DataFrame(all_scores)
    
    if df.empty:
        print("ERROR: Tidak ada hasil prediksi yang berhasil diproses. Plotting dibatalkan.")
        return
        
    df.dropna(inplace=True) # Hapus kelas yang bernilai NaN (struktur anatomis absen)
    
    # 1. Simpan metrik ke CSV
    csv_out = os.path.join(out_plot_dir, f"test_metrics_Exp{args.dataset}_Fold{args.fold}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Metrik mentah tersimpan di: {csv_out}")
    
    # 2. Gambar Boxplot Dice per Kelas
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.boxplot(x="Class", y="Dice", hue="Class", data=df, palette="Set2", legend=False)
    sns.stripplot(x="Class", y="Dice", data=df, color=".25", alpha=0.5)
    
    plt.title(f"Performa Dice (Held-Out Test Set) - Eksperimen {args.dataset} (Fold {args.fold})")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Dice Similarity Coefficient (DSC)")
    
    plot_out = os.path.join(out_plot_dir, f"boxplot_Exp{args.dataset}_Fold{args.fold}.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot tersimpan di: {plot_out}")
    
    # 3. Hitung Rata-rata
    mean_dice = df.groupby("Class")["Dice"].mean().reset_index()
    print("\nRATA-RATA DICE (Test Set):")
    print(mean_dice.to_string(index=False))

if __name__ == "__main__":
    main()
