import os
import sys
import gc
import cv2
import numpy as np
import pandas as pd
import shutil
import yaml
import random
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

# ==========================================
# 0. Setup and Resource Management
# ==========================================
# Clear GPU cache before starting
gc.collect()
torch.cuda.empty_cache()

# Generate Output Directories based on Timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PROJECT_DIR = '/home/ubuntu/train/Result_Analysis'
RUN_DIR = os.path.join(BASE_PROJECT_DIR, f'Run_Master_{TIMESTAMP}')
LOG_FILE_PATH = os.path.join(RUN_DIR, 'Execution_Log.txt')
CHECK_OUTPUT_DIR = os.path.join(RUN_DIR, 'Visual_Checks')
TRAIN_OUTPUT_DIR = os.path.join(RUN_DIR, 'Training')

# Temporary Dataset Directory for YOLO
YOLO_DATASET_DIR = '/home/ubuntu/train/YOLO_Dataset_Master'

# Ensure directories exist
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECK_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

# Logger Class to write to both Terminal and File
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout and stderr to the logger
sys.stdout = DualLogger(LOG_FILE_PATH)
sys.stderr = sys.stdout

print(f"[INFO] --- STARTING MASTER YOLO SESSION: {TIMESTAMP} ---")
print(f"[INFO] Hardware Detected: Optimization for RTX 3080Ti (12GB)")
print(f"[INFO] Strategy: Hybrid Approach (Deep Learning + Texture Analysis)")

# ==========================================
# 1. Configuration (Hyperparameters)
# ==========================================
# ==========================================
# 1. Configuration (Hyperparameters)
# ==========================================
CONFIG = {
    'raw_dir': '/home/ubuntu/train/raw',
    'mask_dir': '/home/ubuntu/train/Segmentation_Masking1',
    
    # مدل را روی Large نگه دارید (چون دقت برایتان مهم است)
    'model_type': 'yolo11l-seg.pt',   
    
    # --- تغییر کلیدی 1: کاهش رزولوشن به استاندارد High-Res ---
    # این کار باعث می‌شود مصرف رم به شدت کاهش یابد و جا برای تصاویر بیشتر باز شود
    'img_size': 1024,       
    
    # --- تغییر کلیدی 2: افزایش بچ سایز ---
    # با رزولوشن 1024، شما احتمالا می‌توانید 8 تصویر را در 11 گیگ جا دهید.
    # اگر باز هم ارور داد، سیستم خودش روی 4 یا 6 تنظیم می‌کند که عالی است.
    'batch_size': 8,        
    
    'workers': 8,           
    'epochs': 200,          
    'patience': 40,         
    
    'noise_prob': 0.3,  
    'min_area': 50,       
    'texture_variance_threshold': 150.0, 
}

#========================================
# 2. Image Processing & Texture Helper Functions
# ==========================================
def add_gaussian_noise(image):
    """Adds Gaussian noise to simulate dirt/dust."""
    row, col = image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col)).astype('float32')
    noisy = image.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def analyze_texture(roi):
    """
    Analyzes texture to distinguish Nematodes (Granular) from Artifacts (Smooth bubbles/threads).
    Returns the variance of the Laplacian.
    """
    if roi.size == 0: return 0
    
    # Laplacian highlights edges and texture roughness
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    score = np.var(laplacian)
    return score

# ==========================================
# 3. Dataset Preparation
# ==========================================
def prepare_yolo_dataset():
    print(f"\n[INFO] Preparing YOLO Dataset Structure...")
    
    # Safely remove existing dataset directory
    if os.path.exists(YOLO_DATASET_DIR): 
        try:
            shutil.rmtree(YOLO_DATASET_DIR)
        except OSError as e:
            print(f"[WARNING] Could not delete folder immediately: {e}")
            # Try renaming as a fallback to avoid locking issues
            os.rename(YOLO_DATASET_DIR, YOLO_DATASET_DIR + f"_trash_{int(time.time())}")

    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', split), exist_ok=True)

    # Gather images
    all_files = os.listdir(CONFIG['raw_dir'])
    all_images = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif', '.tiff'))]
    
    data_entries = []
    
    print(f"[INFO] Found {len(all_images)} images. Matching with masks...")

    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        
        # Search for corresponding mask with various extensions/prefixes
        possible_mask_names = [
            f"contour_{base_name}.png", f"contour_{base_name}.jpg", f"contour_{base_name}.tif",
            f"{base_name}.png", f"{base_name}.tif"
        ]
        
        found_mask = None
        for cand in possible_mask_names:
            cand_path = os.path.join(CONFIG['mask_dir'], cand)
            if os.path.exists(cand_path):
                found_mask = cand_path
                break
        
        entry = {
            'img_path': os.path.join(CONFIG['raw_dir'], img_name),
            'mask_path': found_mask, 
            'is_negative': (found_mask is None)
        }
        data_entries.append(entry)
        
    # Split Dataset
    train, test_val = train_test_split(data_entries, test_size=0.2, random_state=42)
    val, test = train_test_split(test_val, test_size=0.5, random_state=42)
    
    datasets = {'train': train, 'val': val, 'test': test}
    
    # CLAHE for contrast enhancement (crucial for nematode texture)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    for split, data in datasets.items():
        print(f"   -> Processing {split}: {len(data)} images...")
        for entry in data:
            img = cv2.imread(entry['img_path'], cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # Resize image
            img_resized = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
            
            # Prepare Mask
            if entry['is_negative']:
                mask_resized = np.zeros_like(img_resized)
            else:
                mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask_resized = np.zeros_like(img_resized)
                else:
                    mask_resized = cv2.resize(mask, (CONFIG['img_size'], CONFIG['img_size']), interpolation=cv2.INTER_NEAREST)
                    _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            
            # Function to save Image/Label pair
            def save_pair(image_data, mask_data, suffix=""):
                contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                yolo_lines = []
                
                for cnt in contours:
                    if cv2.contourArea(cnt) > CONFIG['min_area']:
                        # Polygon simplification
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        
                        if len(approx) < 3: continue
                        approx = approx.reshape(-1, 2)
                        
                        # Normalize coordinates (0-1)
                        norm = approx.astype(float) / CONFIG['img_size']
                        norm = np.clip(norm, 0.0, 1.0) # Clip to ensure valid YOLO format
                        
                        # Class 0 for Nematode
                        line = "0 " + " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in norm])
                        yolo_lines.append(line)
                
                base_fname = os.path.splitext(os.path.basename(entry['img_path']))[0]
                fname = base_fname + suffix
                
                # Save Image
                cv2.imwrite(os.path.join(YOLO_DATASET_DIR, 'images', split, fname + ".jpg"), image_data)
                
                # Save Label (Create empty file even if no nematode found)
                with open(os.path.join(YOLO_DATASET_DIR, 'labels', split, fname + ".txt"), 'w') as f:
                    for l in yolo_lines: f.write(l + "\n")

            # Apply CLAHE and save original
            img_clahe = clahe.apply(img_resized)
            save_pair(img_clahe, mask_resized)
            
            # Augmentation (Train only)
            if split == 'train':
                # Noise Injection
                if random.random() < CONFIG['noise_prob']:
                    noisy = add_gaussian_noise(img_clahe)
                    save_pair(noisy, mask_resized, "_noisy")
                
                # Flipping (Horizontal)
                if not entry['is_negative'] and random.random() < 0.5:
                    flipped_img = cv2.flip(img_clahe, 1)
                    flipped_mask = cv2.flip(mask_resized, 1)
                    save_pair(flipped_img, flipped_mask, "_flip")

    # Create YAML file
    yaml_content = {
        'path': YOLO_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'nematode'}
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f: yaml.dump(yaml_content, f)
    
    return yaml_path

# ==========================================
# 4. Training (Optimized for RTX 3080Ti)
# ==========================================
def train_yolo(data_yaml):
    print("\n[INFO] --- Initializing YOLO Model ---")
    model = YOLO(CONFIG['model_type']) 
    
    print("[INFO] Starting Training on RTX 3080Ti...")
    results = model.train(
        data=data_yaml,
        project=TRAIN_OUTPUT_DIR,
        name='nematode_3080ti_run',
        
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        patience=CONFIG['patience'],
        
        # Optimization settings
        optimizer='AdamW',    # Better for segmentation
        lr0=0.001,           
        cos_lr=True,          
        
        nbs=64,               # Nominal batch size
        
        # Mask settings
        overlap_mask=True,    
        mask_ratio=4,         # High resolution masks (3080Ti can handle this)
        
        # Augmentations
        degrees=180.0,        # Full rotation allowed
        translate=0.1,
        scale=0.5,            
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,            
        close_mosaic=10,      
        
        # Hardware settings
        device=0,             # Ensure GPU 0 is used
        workers=CONFIG['workers'],
        verbose=True,
        plots=True,
        amp=True              # Automatic Mixed Precision (Faster on 3000 series)
    )
    return results

# ==========================================
# 5. Hybrid Evaluation (Detection + Texture)
# ==========================================
def evaluate_and_report(model_path):
    print("\n[INFO] --- Starting Hybrid Evaluation (Detection + Texture Analysis) ---")
    
    best_model = YOLO(model_path)
    test_img_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'test')
    test_lbl_dir = os.path.join(YOLO_DATASET_DIR, 'labels', 'test')
    
    test_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    if not test_files: 
        print("[WARNING] No test images found.")
        return

    print("[INFO] Predicting with Test-Time Augmentation (TTA)...")
    # TTA helps reduce random noise predictions
    preds = best_model.predict(
        source=test_img_dir, 
        conf=0.25, 
        iou=0.45, 
        augment=True,       # TTA enabled
        retina_masks=True,  # Use high-res masks
        stream=True,
        device=0
    )
    
    excel_data = []
    
    for result in preds:
        filename = os.path.basename(result.path)
        base_name = os.path.splitext(filename)[0]
        
        orig_img = cv2.imread(result.path)
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) # For texture analysis
        h, w, _ = orig_img.shape
        
        # --- 1. Draw Ground Truth ---
        gt_img = orig_img.copy()
        lbl_path = os.path.join(test_lbl_dir, base_name + ".txt")
        gt_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    gt_count += 1
                    coords = np.array(list(map(float, line.strip().split()))[1:]).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    cv2.polylines(gt_img, [coords.astype(np.int32)], True, (0, 255, 0), 2)
        
        # --- 2. Process Predictions with Texture Filtering ---
        pred_img = orig_img.copy()
        detected_areas = []
        filtered_count = 0 
        
        if result.masks is not None:
            # Get masks data
            masks_data = result.masks.data.cpu().numpy()
            
            # Resize masks to match original image dimensions if necessary
            if masks_data.shape[1:] != (h, w):
                masks_data = [cv2.resize(m, (w, h)) for m in masks_data]
            
            for mask_bitmap in masks_data:
                # Binarize mask
                mask_bin = (mask_bitmap > 0.5).astype(np.uint8)
                
                # Find contour of the predicted mask
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                cnt = contours[0] 
                
                area = cv2.contourArea(cnt)
                if area < CONFIG['min_area']: continue
                
                # --- Hybrid Texture Analysis ---
                # Mask the original gray image to get only the nematode body
                masked_roi = cv2.bitwise_and(gray_img, gray_img, mask=mask_bin)
                
                # Crop to bounding box to save processing time
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                roi_crop = masked_roi[y:y+h_box, x:x+w_box]
                
                # Only analyze non-zero pixels (ignore black background)
                valid_pixels = roi_crop[roi_crop > 0]
                
                is_nematode = True
                texture_score = 0
                
                if len(valid_pixels) > 0:
                    # Calculate texture score
                    texture_score = analyze_texture(roi_crop)
                    
                    # Filter Logic: Smooth texture = Bubble/Artifact
                    if texture_score < CONFIG['texture_variance_threshold']:
                        is_nematode = False
                        filtered_count += 1
                        # Draw Yellow (Rejected)
                        cv2.drawContours(pred_img, [cnt], -1, (0, 255, 255), 1) 
                    else:
                        # Confirmed Nematode
                        detected_areas.append(area)
                        # Draw Red (Accepted)
                        cv2.drawContours(pred_img, [cnt], -1, (0, 0, 255), 2)
                        cv2.fillPoly(pred_img, [cnt], (0, 0, 255))
        
        # Overlay predictions with transparency
        pred_img = cv2.addWeighted(orig_img, 0.6, pred_img, 0.4, 0)
        
        # --- 3. Save Visuals ---
        cv2.putText(gt_img, f"GT: {gt_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(pred_img, f"Pred: {len(detected_areas)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(pred_img, f"Filtered: {filtered_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        combined_view = np.hstack((gt_img, pred_img))
        cv2.imwrite(os.path.join(CHECK_OUTPUT_DIR, f"Check_{filename}"), combined_view)
        
        detected_areas.sort(reverse=True)
        row = {
            'Filename': filename, 
            'GT_Count': gt_count, 
            'Pred_Count': len(detected_areas), 
            'Filtered_Artifacts': filtered_count
        }
        for idx, area in enumerate(detected_areas):
            row[f'Area_{idx+1}'] = area
        excel_data.append(row)

    if excel_data:
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(RUN_DIR, 'Final_Report_Hybrid.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"[SUCCESS] Hybrid Analysis Report saved: {excel_path}")
        print(f"[SUCCESS] Visuals saved in: {CHECK_OUTPUT_DIR}")
    else:
        print("[WARNING] No predictions generated.")

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    try:
        # Step 1: Prepare Dataset
        yaml_path = prepare_yolo_dataset()
        
        # Step 2: Train Model
        train_results = train_yolo(yaml_path)
        
        # Step 3: Get Best Model
        # YOLO structure: runs/project/name/weights/best.pt
        best_weights = os.path.join(train_results.save_dir, 'weights', 'best.pt')
        print(f"\n[INFO] Best Weights located at: {best_weights}")
        
        # Step 4: Evaluate
        evaluate_and_report(best_weights)
        
        print(f"\n[DONE] Finished successfully. All outputs in: {RUN_DIR}")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()