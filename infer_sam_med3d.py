import os
import glob
from tqdm import tqdm
import medim
from utils.infer_utils import validate_paired_img_gt

def run_inference():
    img_dir = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/imagesVal"
    gt_dir = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/labelsVal"
    out_dir = "/home/D13K48009/raid/SAM_Dataset/Exp1_Anterior/predVal"
    ckpt_path = "/home/D13K48009/raid/SAM_results/Exp1_Anterior/Sinus_Anterior/sam_model_dice_best.pth"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading SAM-Med3D Model dari {ckpt_path}...")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    
    gt_fname_list = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
    print(f"Ditemukan {len(gt_fname_list)} file untuk diproses.")
    
    for gt_path in tqdm(gt_fname_list):
        case_name = os.path.basename(gt_path).replace(".nii.gz", "")
        img_path = os.path.join(img_dir, f"{case_name}.nii.gz")
        out_path = os.path.join(out_dir, f"{case_name}.nii.gz")
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} tidak ditemukan!")
            continue
            
        # Gunakan 1 point prompt untuk memicu prediksi (sama persis dengan validation originalnya)
        validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
        
    print("Proses Inference selesai!")

if __name__ == "__main__":
    run_inference()
