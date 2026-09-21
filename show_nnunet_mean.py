import json
import os
import sys

def main():
    json_path = "/home/D13K48009/raid/nnUNet_results/Dataset101_SinusExp1/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
    
    if not os.path.exists(json_path):
        print(f"File tidak ditemukan: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if "metrics" not in data or "mean" not in data["metrics"]:
        print("Format summary.json tidak sesuai (tidak ada kunci 'mean').")
        return
        
    mean_metrics = data["metrics"]["mean"]
    
    print("\n=======================================================")
    print("      HASIL RATA-RATA (MEAN) nnU-Net VALIDATION        ")
    print("=======================================================")
    print(f"{'Class':<20} | {'Dice':<8} | {'IoU':<8} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    
    for class_id, metrics in mean_metrics.items():
        if class_id == "all":
            class_name = "ALL (Average)"
        elif class_id == "1":
            class_name = "1 (Anterior)"
        elif class_id == "2":
            class_name = "2 (Posterior)"
        else:
            class_name = f"Class {class_id}"
            
        dice = metrics.get("Dice", 0)
        iou = metrics.get("IoU", 0)
        
        # Calculate Precision and Recall manually if not present, otherwise use if present
        tp = metrics.get("TP", 0)
        fp = metrics.get("FP", 0)
        fn = metrics.get("FN", 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"{class_name:<20} | {dice:.4f}   | {iou:.4f}   | {precision:.4f}     | {recall:.4f}")
        
    print("=======================================================\n")

if __name__ == "__main__":
    main()
