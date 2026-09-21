import os
import glob
import pandas as pd
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from monai.metrics import (
    DiceMetric, 
    MeanIoU, 
    HausdorffDistanceMetric, 
    SurfaceDistanceMetric
)

def compute_binary_metrics():
    # Paths for Anterior class
    gt_dir = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/labelsVal"
    pred_dir = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/predVal"
    output_csv = "/home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results/sam_metrics_table.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
    
    if len(gt_files) == 0:
        print("Tidak ada file Ground Truth di folder labelsVal!")
        return

    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    asd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")

    results = []
    
    print(f"Menghitung metrik untuk {len(gt_files)} pasien...")
    for gt_path in tqdm(gt_files):
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediksi untuk {filename} tidak ditemukan.")
            continue
            
        # Load images
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        
        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)
        
        # Ensure binary [0, 1]
        gt_arr = (gt_arr > 0).astype(np.float32)
        pred_arr = (pred_arr > 0).astype(np.float32)
        
        # Convert to torch tensor with batch and channel dims [B, C, D, H, W]
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_arr).unsqueeze(0).unsqueeze(0)
        
        # Compute metrics
        dice = dice_metric(y_pred=pred_tensor, y=gt_tensor).item()
        iou = iou_metric(y_pred=pred_tensor, y=gt_tensor).item()
        
        # HD95 and ASD can be NaN or inf if mask is empty, handle carefully
        try:
            hd95 = hd95_metric(y_pred=pred_tensor, y=gt_tensor).item()
        except:
            hd95 = np.nan
            
        try:
            asd = asd_metric(y_pred=pred_tensor, y=gt_tensor).item()
        except:
            asd = np.nan
            
        # Calculate Precision manually (TP / (TP + FP))
        TP = ((pred_arr == 1) & (gt_arr == 1)).sum()
        FP = ((pred_arr == 1) & (gt_arr == 0)).sum()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
        results.append({
            "Case": filename.replace(".nii.gz", ""),
            "Dice": round(dice, 4),
            "IoU": round(iou, 4),
            "Precision": round(precision, 4),
            "HD95 (mm)": round(hd95, 4) if not np.isnan(hd95) else "N/A",
            "ASD (mm)": round(asd, 4) if not np.isnan(asd) else "N/A"
        })
        
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Calculate Mean
    mean_row = {
        "Case": "MEAN",
        "Dice": round(df["Dice"].mean(), 4),
        "IoU": round(df["IoU"].mean(), 4),
        "Precision": round(df["Precision"].mean(), 4),
    }
    
    # Safely calculate mean for HD95 and ASD
    hd95_vals = [r["HD95 (mm)"] for r in results if r["HD95 (mm)"] != "N/A"]
    asd_vals = [r["ASD (mm)"] for r in results if r["ASD (mm)"] != "N/A"]
    
    mean_row["HD95 (mm)"] = round(sum(hd95_vals)/len(hd95_vals), 4) if hd95_vals else "N/A"
    mean_row["ASD (mm)"] = round(sum(asd_vals)/len(asd_vals), 4) if asd_vals else "N/A"
    
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n=========================================")
    print(f"Kalkulasi Selesai!")
    print(f"Tabel Metrik berhasil disimpan di: {output_csv}")
    print(f"=========================================\n")
    print(df.tail(1)) # Print Mean row

if __name__ == "__main__":
    compute_binary_metrics()
