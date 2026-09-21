#!/usr/bin/env python3
"""
Standalone SAM-Med3D Evaluation Script
---------------------------------------
Melakukan Inference + Kalkulasi Metrik dalam SATU skrip.
Tidak membutuhkan: medim, torchio, pandas, atau monai.
Hanya butuh: torch, nibabel, numpy, scipy, tqdm.
"""
import sys
import os
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm

# Add SAM-Med3D to Python path
SAM_DIR = "/home/D13K48009/raid/Clara/sinus-ct-train/SAM-Med3D"
sys.path.insert(0, SAM_DIR)

# ==================== CONFIGURATION ====================
CKPT_PATH  = "/home/D13K48009/raid/SAM_results/Exp1_Anterior/Sinus_Anterior/sam_model_dice_best.pth"
IMG_DIR    = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/imagesVal"
GT_DIR     = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/labelsVal"
OUTPUT_CSV = "/home/D13K48009/raid/SAM_results/Exp1_Anterior/Test_Results/sam_metrics_table.csv"
TARGET_SIZE = (128, 128, 128)
# ========================================================

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)

def preprocess_image(image):
    """Normalize to [0,1] and resize to 128^3"""
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=TARGET_SIZE, mode='trilinear', align_corners=False)
    return resized

def preprocess_label(label):
    """Binarize and resize to 128^3"""
    label = (label > 0).astype(np.float32)
    tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=TARGET_SIZE, mode='nearest')
    return resized

def get_center_point(mask_3d):
    """Get center of mass of binary mask as point prompt [z, y, x]"""
    coords = np.where(mask_3d > 0.5)
    if len(coords[0]) == 0:
        return None
    center = [int(np.mean(c)) for c in coords]
    return center

def compute_surface_distances(pred_bin, gt_bin):
    """Compute HD95 and ASD using scipy"""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    
    pred_border = pred_bin ^ ndimage.binary_erosion(pred_bin, iterations=1)
    gt_border   = gt_bin  ^ ndimage.binary_erosion(gt_bin, iterations=1)
    
    if not pred_border.any() or not gt_border.any():
        return None, None
    
    pred_coords = np.array(np.where(pred_border)).T.astype(np.float64)
    gt_coords   = np.array(np.where(gt_border)).T.astype(np.float64)
    
    # Subsample for speed if too many surface voxels
    MAX_PTS = 5000
    if len(pred_coords) > MAX_PTS:
        idx = np.random.choice(len(pred_coords), MAX_PTS, replace=False)
        pred_coords = pred_coords[idx]
    if len(gt_coords) > MAX_PTS:
        idx = np.random.choice(len(gt_coords), MAX_PTS, replace=False)
        gt_coords = gt_coords[idx]
    
    tree_gt   = cKDTree(gt_coords)
    tree_pred = cKDTree(pred_coords)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_coords)
    dist_gt_to_pred, _ = tree_pred.query(gt_coords)
    
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = float(np.percentile(all_dists, 95))
    asd  = float(np.mean(all_dists))
    return hd95, asd

def run_inference(model, image_tensor, point_3d, device):
    """Run SAM-Med3D inference with a single point prompt"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Image embedding
        image_embedding = model.image_encoder(image_tensor)
        
        # Point prompt: [B, N_points, 3]
        point_coords = torch.tensor([[point_3d]], dtype=torch.float32).to(device)
        point_labels = torch.tensor([[1]], dtype=torch.long).to(device)  # 1 = foreground
        
        # Prompt encoding
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # Mask decoding
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
        pred = (torch.sigmoid(low_res_masks) > 0.5).float()
        return pred.cpu()

def load_sam_model(ckpt_path, device):
    """Try loading SAM-Med3D model with multiple registry keys"""
    from segment_anything.build_sam3D import sam_model_registry3D
    
    for key in ["vit_b_ori", "vit_b"]:
        try:
            print(f"  Trying model key '{key}'...")
            sam_model = sam_model_registry3D[key](checkpoint=None)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sam_model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                sam_model.load_state_dict(ckpt["state_dict"])
            else:
                sam_model.load_state_dict(ckpt)
            
            sam_model.to(device)
            sam_model.eval()
            print(f"  Model loaded successfully with key '{key}'!")
            return sam_model
        except Exception as e:
            print(f"  Key '{key}' failed: {e}")
            continue
    
    raise RuntimeError("Gagal memuat model SAM-Med3D dengan semua key yang tersedia.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # ===== Step 1: Check validation data =====
    gt_files = sorted(glob(os.path.join(GT_DIR, "*.nii.gz")))
    if len(gt_files) == 0:
        print(f"ERROR: Tidak ada file .nii.gz di {GT_DIR}")
        print("Pastikan Anda sudah menjalankan: python split_sam_validation.py")
        return
    print(f"Ditemukan {len(gt_files)} kasus validasi di {GT_DIR}")
    
    # ===== Step 2: Load model =====
    print(f"\nMemuat model dari {CKPT_PATH}...")
    sam_model = load_sam_model(CKPT_PATH, device)
    
    # ===== Step 3: Inference + Metrics =====
    results = []
    print(f"\nMemulai Inference & Kalkulasi Metrik...\n")
    
    for gt_path in tqdm(gt_files, desc="Evaluating"):
        fname     = os.path.basename(gt_path)
        case_name = fname.replace(".nii.gz", "")
        img_path  = os.path.join(IMG_DIR, fname)
        
        if not os.path.exists(img_path):
            print(f"  Warning: {fname} tidak ditemukan di {IMG_DIR}, skip.")
            continue
        
        # Load
        image = load_nifti(img_path)
        gt    = load_nifti(gt_path)
        
        # Preprocess
        img_tensor = preprocess_image(image)
        gt_tensor  = preprocess_label(gt)
        gt_resized = gt_tensor.squeeze().numpy()
        
        # Point prompt from GT center
        center = get_center_point(gt_resized)
        if center is None:
            print(f"  Warning: {case_name} memiliki mask kosong, skip.")
            continue
        
        # Inference
        try:
            pred_tensor = run_inference(sam_model, img_tensor, center, device)
            pred = pred_tensor.squeeze().numpy()
        except Exception as e:
            print(f"  Inference error ({case_name}): {e}")
            continue
        
        # === Compute ALL metrics ===
        pred_bin = pred > 0.5
        gt_bin   = gt_resized > 0.5
        
        TP = int(np.sum(pred_bin & gt_bin))
        FP = int(np.sum(pred_bin & ~gt_bin))
        FN = int(np.sum(~pred_bin & gt_bin))
        TN = int(np.sum(~pred_bin & ~gt_bin))
        
        dice      = (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0
        iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        
        # Surface distances (HD95, ASD)
        try:
            hd95, asd = compute_surface_distances(pred_bin, gt_bin)
        except Exception:
            hd95, asd = None, None
        
        row = {
            "Case":      case_name,
            "Dice":      round(dice, 4),
            "IoU":       round(iou, 4),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "Accuracy":  round(accuracy, 4),
            "HD95":      round(hd95, 4) if hd95 is not None else "N/A",
            "ASD":       round(asd, 4) if asd is not None else "N/A",
        }
        results.append(row)
        print(f"  {case_name}: Dice={dice:.4f} | IoU={iou:.4f} | Prec={precision:.4f} | Recall={recall:.4f}")
    
    if not results:
        print("\nTidak ada hasil yang dihitung. Periksa folder imagesVal dan labelsVal Anda.")
        return
    
    # ===== Step 4: Compute Mean =====
    n = len(results)
    mean_row = {
        "Case":      "MEAN",
        "Dice":      round(sum(r["Dice"]      for r in results) / n, 4),
        "IoU":       round(sum(r["IoU"]       for r in results) / n, 4),
        "Precision": round(sum(r["Precision"] for r in results) / n, 4),
        "Recall":    round(sum(r["Recall"]    for r in results) / n, 4),
        "Accuracy":  round(sum(r["Accuracy"]  for r in results) / n, 4),
    }
    hd95_vals = [r["HD95"] for r in results if r["HD95"] != "N/A"]
    asd_vals  = [r["ASD"]  for r in results if r["ASD"]  != "N/A"]
    mean_row["HD95"] = round(sum(hd95_vals)/len(hd95_vals), 4) if hd95_vals else "N/A"
    mean_row["ASD"]  = round(sum(asd_vals)/len(asd_vals), 4)   if asd_vals  else "N/A"
    results.append(mean_row)
    
    # ===== Step 5: Save CSV =====
    fieldnames = ["Case", "Dice", "IoU", "Precision", "Recall", "Accuracy", "HD95", "ASD"]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI SAM-Med3D (Anterior Sinus)")
    print(f"{'='*60}")
    print(f"  Jumlah Kasus   : {n}")
    print(f"  Mean Dice      : {mean_row['Dice']}")
    print(f"  Mean IoU       : {mean_row['IoU']}")
    print(f"  Mean Precision : {mean_row['Precision']}")
    print(f"  Mean Recall    : {mean_row['Recall']}")
    print(f"  Mean Accuracy  : {mean_row['Accuracy']}")
    print(f"  Mean HD95      : {mean_row['HD95']}")
    print(f"  Mean ASD       : {mean_row['ASD']}")
    print(f"{'='*60}")
    print(f"  Tabel CSV tersimpan di: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
