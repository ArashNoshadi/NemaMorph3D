import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# ==========================================
# 1. تنظیمات مسیرها (اینجا را ویرایش کنید)
# ==========================================
# مسیر فعلی تصاویر خام و ماسک‌ها
RAW_IMAGE_DIR = '/mnt/g/Paper/Eggplant/Image/10x/raw'
MASK_DIR = '/mnt/g/Paper/Eggplant/Image/10x/pre_processing/Segmentation_Masking1'

# مسیری که می‌خواهید دیتاست آماده شده برای YOLO آنجا ذخیره شود
OUTPUT_DIR = '/mnt/g/Paper/Eggplant/Image/10x/YOLO_Dataset'

# درصد داده‌های تست (مثلاً 0.2 یعنی 20 درصد برای اعتبار سنجی)
VAL_SIZE = 0.2

# اگر پیشوند خاصی در نام فایل‌های ماسک وجود دارد (مثلاً contour_image.png)
# اگر نام فایل‌ها دقیقاً یکی است، این را خالی بگذارید: ""
MASK_PREFIX = "contour_" 

# حداقل مساحت برای نادیده گرفتن آشغال‌های ریز (به پیکسل)
MIN_AREA_THRESHOLD = 50 

# ==========================================
# 2. توابع کمکی
# ==========================================

def create_folders(base_path):
    """ساخت ساختار پوشه‌های استاندارد YOLO"""
    if os.path.exists(base_path):
        print(f"[WARNING] Output directory {base_path} already exists.")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)

def find_mask_file(image_name, mask_dir, prefix):
    """پیدا کردن فایل ماسک متناظر با تصویر"""
    # حذف پسوند فایل (مثلاًimage.tif می‌شود image)
    base_name = os.path.splitext(image_name)[0]
    
    # لیست نام‌های احتمالی برای ماسک (فرمت‌های رایج ماسک)
    # اینجا هم tif اضافه شد محض احتیاط
    extensions = [".png", ".jpg", ".bmp", ".tif", ".tiff"]
    possible_names = []

    # حالت 1: نام ماسک دقیقاً مثل نام تصویر است
    for ext in extensions:
        possible_names.append(base_name + ext)
    
    # حالت 2: نام ماسک پیشوند دارد (مثل contour_file.png)
    if prefix:
        for ext in extensions:
            possible_names.append(prefix + base_name + ext)
        
    for m_name in possible_names:
        m_path = os.path.join(mask_dir, m_name)
        if os.path.exists(m_path):
            return m_path
    return None

def mask_to_yolo_polygon(mask_path, img_width, img_height):
    """تبدیل ماسک باینری به مختصات پلی‌گون نرمال شده"""
    # خواندن ماسک (حتی اگر tif باشد)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"[WARNING] Could not read mask: {mask_path}")
        return []

    # اطمینان از باینری بودن کامل
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # پیدا کردن کانتورها (مرز اشیاء)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD:
            # ساده‌سازی کانتور برای کاهش تعداد نقاط (بهینه‌سازی سرعت آموزش)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # اگر شکل خیلی کوچک شد (کمتر از 3 نقطه)، نادیده بگیر
            if len(approx) < 3:
                continue
            
            # نرمال‌سازی مختصات (بین 0 و 1)
            normalized_points = []
            for point in approx:
                x, y = point[0]
                # محدود کردن مختصات به بازه [0, 1] برای جلوگیری از خطای YOLO
                x_norm = min(max(x / img_width, 0), 1)
                y_norm = min(max(y / img_height, 0), 1)
                
                normalized_points.append(x_norm)
                normalized_points.append(y_norm)
            
            polygons.append(normalized_points)
            
    return polygons

# ==========================================
# 3. اجرای اصلی
# ==========================================

def process_dataset():
    print("[INFO] Starting Dataset Conversion for YOLOv11-seg...")
    
    # دریافت لیست تصاویر - ویرایش شده برای TIF
    all_files = os.listdir(RAW_IMAGE_DIR)
    images = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
    
    if len(images) == 0:
        print(f"[ERROR] No .tif or .tiff images found in {RAW_IMAGE_DIR}")
        return

    print(f"[INFO] Found {len(images)} TIF images.")

    # جفت کردن تصاویر و ماسک‌ها
    valid_pairs = []
    missing_masks = 0
    
    for img_name in images:
        mask_path = find_mask_file(img_name, MASK_DIR, MASK_PREFIX)
        if mask_path:
            valid_pairs.append((os.path.join(RAW_IMAGE_DIR, img_name), mask_path))
        else:
            missing_masks += 1
    
    print(f"[INFO] Found {len(valid_pairs)} valid image-mask pairs.")
    if missing_masks > 0:
        print(f"[WARNING] {missing_masks} images did not have a matching mask.")
    
    if len(valid_pairs) == 0:
        print("[ERROR] No matching pairs found. Check paths and prefixes.")
        return

    # تقسیم به Train و Validation
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=VAL_SIZE, random_state=42)
    
    create_folders(OUTPUT_DIR)
    
    datasets = {
        'train': train_pairs,
        'val': val_pairs
    }
    
    for split, data in datasets.items():
        print(f"\nProcessing {split} data ({len(data)} images)...")
        
        for img_path, mask_path in tqdm(data):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            # خواندن ابعاد تصویر برای نرمال‌سازی
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read image {img_path}. Skipping.")
                continue
                
            h, w, _ = img.shape
            
            # 1. کپی تصویر به پوشه images
            # نکته: YOLO معمولا فرمت TIF را ساپورت می‌کند، اما اگر فایل‌ها خیلی حجیم هستند
            # یا 16 بیتی هستند، بهتر است به JPG تبدیل شوند. 
            # این کد فعلا فایل اصلی را کپی می‌کند.
            dest_img_path = os.path.join(OUTPUT_DIR, 'images', split, filename)
            shutil.copy(img_path, dest_img_path)
            
            # 2. تبدیل ماسک به تکست فایل YOLO
            polygons = mask_to_yolo_polygon(mask_path, w, h)
            
            # اگر هیچ پلی‌گانی پیدا نشد (ماسک خالی)، فایل لیبل خالی ساخته می‌شود (که صحیح است)
            label_file_path = os.path.join(OUTPUT_DIR, 'labels', split, name_no_ext + ".txt")
            
            with open(label_file_path, 'w') as f:
                for poly in polygons:
                    # فرمت YOLO Segment: <class_id> <x1> <y1> <x2> <y2> ...
                    # کلاس نماتد را 0 در نظر می‌گیریم
                    line = "0 " + " ".join([f"{p:.6f}" for p in poly])
                    f.write(line + "\n")
    
    # ساخت فایل data.yaml
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'nematode'
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"\n[SUCCESS] Dataset preparation complete!")
    print(f"Data saved in: {OUTPUT_DIR}")
    print(f"YAML config created at: {os.path.join(OUTPUT_DIR, 'data.yaml')}")

if __name__ == "__main__":
    process_dataset()