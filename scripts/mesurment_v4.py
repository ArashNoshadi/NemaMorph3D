import os
import cv2
import numpy as np
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from ultralytics import YOLO
from scipy.spatial.distance import cdist 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import splprep, splev  
from matplotlib.ticker import MaxNLocator
import random

# ==========================================================
# 0. تنظیمات کالیبراسیون و حجم
# ==========================================================
KNOWN_PIXELS = 1030.855
KNOWN_MICRONS = 10
SCALE_FACTOR = KNOWN_MICRONS / KNOWN_PIXELS  
PIXEL_AREA_FACTOR = SCALE_FACTOR ** 2        
VOXEL_VOLUME_FACTOR = SCALE_FACTOR ** 3  # میکرومتر مکعب بر پیکسل مکعب

SHAPE_THRESHOLD = 3.0  
ANDRASSY_C = 1.7       

print(f"=== Calibration & Settings Info ===")
print(f"1 Pixel = {SCALE_FACTOR:.6f} Micrometers")
print(f"1 Voxel (3D Pixel) = {VOXEL_VOLUME_FACTOR:.8f} Micrometers^3")
print("="*40)

# ==========================================================
# 1. تنظیمات مسیرها
# ==========================================================
RAW_IMAGE_DIR = r'G:\Paper\Eggplant\Image\10x\raw' 
MODEL_PATH = r'G:\Paper\Eggplant\Image\10x\try3\Server\pc\Result_Analysis\Run_Master_20260218_095328\Training\nematode_3080ti_run\weights\best.pt'
BASE_OUTPUT_DIR = r'G:\Paper\Eggplant\Image\10x\try3\Server\pc\Measurements5'

EXCEL_OUTPUT = os.path.join(BASE_OUTPUT_DIR, 'Nematode_Morphology_Report.xlsx')
CHECK_DIR = os.path.join(BASE_OUTPUT_DIR, 'Visual_Checks')
VISUAL_3D_DIR = os.path.join(BASE_OUTPUT_DIR, '3D_Publication_Ready')

os.makedirs(CHECK_DIR, exist_ok=True)
os.makedirs(VISUAL_3D_DIR, exist_ok=True)

MAX_3D_EXAMPLES = 317
examples_3d_generated = 0

# ==========================================================
# 2. توابع محاسبات بیولوژیکی و 3 بعدی
# ==========================================================
def generate_3d_z_map(mask_roi, shape_type):
    """
    تولید نقشه ارتفاعی (Z) با مدل هیدرواستاتیک (ضد چاله و فرورفتگی)
    تضمین می‌کند که بدن نماتد به عنوان یک کپسول تحت فشار، کاملاً صاف و بدون گودی باشد.
    """
    # 1. نرم‌کردن ماسک پایه برای حذف خوردگی‌های لبه (Segmentation Bites)
    # این کار مانع از انتقال نویز لبه به نقشه 3D می‌شود
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_smooth)
    
    # 2. محاسبه فاصله هر پیکسل تا لبه نماتد
    dt = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    
    # 3. استخراج اسکلت مرکزی (ستون فقرات)
    img_skel = mask_clean.copy()
    skel = np.zeros(img_skel.shape, np.uint8)
    kernel_skel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img_skel, kernel_skel)
        temp = cv2.dilate(eroded, kernel_skel)
        temp = cv2.subtract(img_skel, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_skel = eroded.copy()
        if cv2.countNonZero(img_skel) == 0:
            break
            
    # 4. محاسبه فاصله تا اسکلت مرکزی
    if cv2.countNonZero(skel) == 0:
        dt_skel = np.zeros_like(dt)
    else:
        skel_inv = cv2.bitwise_not(skel)
        dt_skel = cv2.distanceTransform(skel_inv, cv2.DIST_L2, 5)
    
    # 5. فرمول هندسی نیم‌کره مقطعی
    inside_sqrt = dt * (dt + 2 * dt_skel)
    Z = np.sqrt(np.maximum(inside_sqrt, 0))
    
    # =======================================================
    # 6. اعمال فیلتر فشار هیدرواستاتیک (پر کردن چاله‌های سه‌بعدی)
    # =======================================================
    Z = np.float32(Z)
    
    # این فیلتر مورفولوژیک خاکستری، مثل بتونه تمام فرورفتگی‌ها و گودی‌های روی بدن را پر و هم‌سطح می‌کند
    # هر چاله‌ای کوچکتر از 15 پیکسل کاملاً صاف می‌شود
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    Z = cv2.morphologyEx(Z, cv2.MORPH_CLOSE, kernel_fill)
    
    # 7. تنظیمات نهایی و رفع نویز بر اساس شکل بیولوژیک
    if shape_type == 'Elongated':
        # پوست صیقلی برای نماتدهای کشیده
        Z = cv2.GaussianBlur(Z, (11, 11), 0)
    else:
        # برای سیست‌ها و گلابی‌شکل‌ها حجم بیشتر و صاف‌شدگی عمیق‌تر
        Z = Z * 1.15 
        Z = cv2.GaussianBlur(Z, (15, 15), 0)

    # 8. برگرداندن به ابعاد ماسک اصلی (تا لبه‌های اضافی که در اثر بلور پخش شده‌اند کات شوند)
    Z_final = Z * (mask_roi > 0)
    return Z_final

def get_morphology_metrics(mask, contour, scale, area_scale, voxel_scale, gray_roi):
    """
    استخراج ویژگی‌های پیشرفته مورفولوژیک، بافت و بیومکانیک (سطح Q1)
    """
    # 1. ویژگی‌های پایه
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    real_area = area_px * area_scale
    real_perimeter = perimeter_px * scale
    
    # 2. نقشه فاصله و ابعاد
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    W_px = max_val * 2  
    
    # =============================================================
    # محاسبه طول: مسیر مرکزی توپولوژیک با تبدیل به منحنی نرم (Spline Curve)
    # =============================================================
    pad = 5
    x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
    y1, y2 = max(0, y_b - pad), min(mask.shape[0], y_b + h_b + pad)
    x1, x2 = max(0, x_b - pad), min(mask.shape[1], x_b + w_b + pad)
    mask_c = mask[y1:y2, x1:x2]

    y_idx, x_idx = np.where(mask_c > 0)
    pts = list(zip(x_idx, y_idx))
    n_pts = len(pts)

    L_px_curved_exact = 0
    final_path = []

    if n_pts > 10:
        pt_to_idx = {pt: i for i, pt in enumerate(pts)}
        
        edges_i = []
        edges_j = []
        weights_un = []
        weights_w = []

        dt_c = cv2.distanceTransform(mask_c, cv2.DIST_L2, 5)
        max_dt = np.max(dt_c) if np.max(dt_c) > 0 else 1.0

        for i, (px, py) in enumerate(pts):
            for dx, dy, d_val in [(-1,-1,1.4142), (0,-1,1.0), (1,-1,1.4142), 
                                  (-1,0,1.0),                 (1,0,1.0), 
                                  (-1,1,1.4142),  (0,1,1.0),  (1,1,1.4142)]:
                nx, ny = px + dx, py + dy
                if (nx, ny) in pt_to_idx:
                    j = pt_to_idx[(nx, ny)]
                    edges_i.append(i)
                    edges_j.append(j)
                    weights_un.append(d_val)
                    
                    penalty = (max_dt - dt_c[ny, nx]) / max_dt
                    cost = d_val * (1.0 + 50.0 * (penalty ** 3))
                    weights_w.append(cost)
                    
        adj_un = csr_matrix((weights_un, (edges_i, edges_j)), shape=(n_pts, n_pts))
        adj_w = csr_matrix((weights_w, (edges_i, edges_j)), shape=(n_pts, n_pts))

        dist_start = dijkstra(adj_un, indices=0, return_predecessors=False, unweighted=False)
        dist_start[dist_start == np.inf] = -1
        head_idx = int(np.argmax(dist_start))

        dist_head = dijkstra(adj_un, indices=head_idx, return_predecessors=False, unweighted=False)
        dist_head[dist_head == np.inf] = -1
        tail_idx = int(np.argmax(dist_head))

        dist_w, pred_w = dijkstra(adj_w, indices=head_idx, return_predecessors=True, unweighted=False)
        
        path_local = []
        curr = tail_idx
        while curr != -9999 and curr >= 0:
            path_local.append(pts[curr])
            if curr == head_idx:
                break
            curr = pred_w[curr]
            
        if len(path_local) > 1:
            path_global = [(px + x1, py + y1) for px, py in path_local]
            
            unique_path = [path_global[0]]
            for p in path_global[1:]:
                if p != unique_path[-1]:
                    unique_path.append(p)
                    
            if len(unique_path) > 3:
                x_coords = [p[0] for p in unique_path]
                y_coords = [p[1] for p in unique_path]
                
                tck, u = splprep([x_coords, y_coords], s=len(unique_path) * 1.5, k=3)
                
                u_new = np.linspace(0, 1.0, max(200, len(unique_path) * 2))
                x_new, y_new = splev(u_new, tck)
                
                smoothed_pts = list(zip(x_new, y_new))
            else:
                smoothed_pts = unique_path
                
            final_path = np.array(smoothed_pts, dtype=np.int32).reshape((-1, 1, 2))
            
            for i in range(len(smoothed_pts)-1):
                p1 = smoothed_pts[i]
                p2 = smoothed_pts[i+1]
                L_px_curved_exact += math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    if L_px_curved_exact == 0:
        hull = cv2.convexHull(contour)
        hull_points = hull[:, 0, :]
        if len(hull_points) >= 2:
            distances = cdist(hull_points, hull_points, metric='euclidean')
            i, j = np.unravel_index(distances.argmax(), distances.shape)
            L_px_curved_exact = distances[i, j]
            pt1 = tuple(hull_points[i])
            pt2 = tuple(hull_points[j])
            final_path = np.array([[pt1], [pt2]], dtype=np.int32)

    L_px = L_px_curved_exact
    # =============================================================

    L = L_px * scale
    W = W_px * scale
    shape_type = 'Elongated' if (W > 0 and (L / W) >= SHAPE_THRESHOLD) else 'Round/Cyst'

    # 3. حجم 3 بعدی و تخمین وزن
    mask_cropped_for_vol = mask[y_b:y_b+h_b, x_b:x_b+w_b]
    Z_map_px = generate_3d_z_map(mask_cropped_for_vol, shape_type)
    
    vol_3d = np.sum(Z_map_px * 2) * voxel_scale
    weight_ug = vol_3d * 1.13e-6  # چگالی 1.13 برای تبدیل به میکروگرم

    # 4. ویژگی‌های پیشرفته جدید
    mean_z = np.mean(Z_map_px[mask_cropped_for_vol > 0]) * scale
    approx_surface_area = (2 * real_area) + (real_perimeter * (mean_z * 2))
    svr = approx_surface_area / vol_3d if vol_3d > 0 else 0

    sphericity = (math.pi**(1/3) * (6 * vol_3d)**(2/3)) / approx_surface_area if approx_surface_area > 0 else 0
    globosity = W / L if L > 0 else 0
    tortuosity = (real_perimeter / 2) / L if L > 0 else 1.0

    if gray_roi is not None:
        internal_pixels = gray_roi[mask_cropped_for_vol > 0]
        texture_heterogeneity = np.std(internal_pixels) if len(internal_pixels) > 0 else 0
    else:
        texture_heterogeneity = 0

    thickness_heterogeneity = np.std(dist_transform[mask > 0]) * scale

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        box_center_x, box_center_y = x_b + w_b/2, y_b + h_b/2
        mass_shift = math.sqrt((cx - box_center_x)**2 + (cy - box_center_y)**2) * scale
        ap_mass_dist = mass_shift / L if L > 0 else 0
    else:
        ap_mass_dist = 0

    allometric_area = real_area / (L**2) if L > 0 else 0

    return {
        'Shape_Class': shape_type,
        'Length(um)': L,
        'Max_Width(um)': W,
        'Area(um2)': real_area,
        'Perimeter(um)': real_perimeter,
        'Volume_3D(um3)': vol_3d,
        'Weight_Estimate(ug)': weight_ug,
        'Surface_to_Volume_Ratio': svr,
        'Sphericity_Index': sphericity,
        'Globosity_Index': globosity,
        'Tortuosity': tortuosity,
        'Cytoplasm_Texture(std)': texture_heterogeneity,
        'Thickness_Heterogeneity': thickness_heterogeneity,
        'Mass_Shift_Index': ap_mass_dist,
        'Allometric_Area': allometric_area,
        'Draw_Data': {
            'Center_Circle': max_loc,
            'Radius': int(max_val),
            'Centerline_Path': final_path,
            'Bounding_Box': (x_b, y_b, w_b, h_b)
        }
    }

def create_publication_3d_plot(img_roi, Z_map_px, scale, save_path):
    h, w = Z_map_px.shape
    X_um = np.arange(0, w) * scale
    Y_um = np.arange(0, h) * scale
    X_um, Y_um = np.meshgrid(X_um, Y_um)
    Z_um = Z_map_px * scale  
    
    fig = plt.figure(figsize=(14, 6), dpi=1200) 
    
    ax1 = fig.add_subplot(1, 2, 1)
    img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title("Original Brightfield Microscopy", fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ls = LightSource(azdeg=315, altdeg=45)
    colormap = cm.magma 
    rgb = ls.shade(Z_um, cmap=colormap, vert_exag=1.5, blend_mode='soft')
    
    surf = ax2.plot_surface(X_um, Y_um, Z_um, facecolors=rgb, linewidth=0, antialiased=True, shade=False, rcount=300, ccount=300)
    ax2.set_box_aspect((np.ptp(X_um), np.ptp(Y_um), np.ptp(Z_um)*2))
    ax2.view_init(elev=45, azim=-60)
    ax2.set_title("3D Topographical Reconstruction", fontsize=12, fontweight='bold', fontfamily='serif')
    
    ax2.set_xlabel('Length (\u03bcm)', fontsize=10, fontfamily='serif')
    ax2.set_ylabel('Width (\u03bcm)', fontsize=10, fontfamily='serif')
    ax2.set_zlabel('Thickness (\u03bcm)', fontsize=10, fontfamily='serif')
    
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)

    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(Z_um)
    cbar = fig.colorbar(m, ax=ax2, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('Radius / Half-Thickness (\u03bcm)', fontfamily='serif')
    
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.tick_params(axis='x', labelsize=8, pad=0)
    ax2.tick_params(axis='y', labelsize=8, pad=0)
    ax2.tick_params(axis='z', labelsize=8, pad=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ==========================================================
# 3. بارگذاری مدل YOLO
# ==========================================================
print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH) 
except Exception as e:
    print(f"[ERROR] Could not load model. Check the path! Error: {e}")
    exit()

image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
    image_files.extend(glob.glob(os.path.join(RAW_IMAGE_DIR, ext)))

print(f"[INFO] Found {len(image_files)} images.")

total_images = len(image_files)
random_3d_indices = set(random.sample(range(total_images), min(MAX_3D_EXAMPLES, total_images)))
print(f"[INFO] Randomly selected {len(random_3d_indices)} images for 3D generation.")

# ==========================================================
# 4. پردازش و اندازه‌گیری
# ==========================================================
flat_data_list = []

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    print(f"[{idx+1}/{len(image_files)}] Analyzing: {filename}")
    
    original_img = cv2.imread(img_path)
    if original_img is None: continue
    h, w, _ = original_img.shape
    
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 
    
    results = model.predict(source=img_path, conf=0.25, iou=0.45, retina_masks=True, verbose=False)
    result = results[0]
    contours_found = []
    
    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy() 
        for i, mask_tensor in enumerate(masks_data):
            if mask_tensor.shape != (h, w):
                mask_tensor = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask_bin = (mask_tensor > 0.5).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 50:
                    isolated_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(isolated_mask, [c], -1, 255, thickness=-1)
                    
                    x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                    gray_roi = gray_img[y_b:y_b+h_b, x_b:x_b+w_b]
                    
                    metrics = get_morphology_metrics(isolated_mask, c, SCALE_FACTOR, PIXEL_AREA_FACTOR, VOXEL_VOLUME_FACTOR, gray_roi)
                    contours_found.append(metrics)
                    
                    # ---------------------------------------------------
                    # تولید پلات 3 بعدی
                    # ---------------------------------------------------
                    if idx in random_3d_indices and examples_3d_generated < MAX_3D_EXAMPLES:
                        pad_3d = 40 
                        x1, y1 = max(0, x_b-pad_3d), max(0, y_b-pad_3d)
                        x2, y2 = min(w, x_b+w_b+pad_3d), min(h, y_b+h_b+pad_3d)
                        
                        img_roi = original_img[y1:y2, x1:x2].copy()
                        roi_mask = isolated_mask[y1:y2, x1:x2]
                        
                        Z_map_roi = generate_3d_z_map(roi_mask, metrics['Shape_Class'])
                        
                        save_name_3d = os.path.join(VISUAL_3D_DIR, f"3D_Render_{filename}_Nem_{i+1}.jpg")
                        create_publication_3d_plot(img_roi, Z_map_roi, SCALE_FACTOR, save_name_3d)
                        
                        examples_3d_generated += 1
                        print(f"   -> Publication 3D Plot Created ({examples_3d_generated}/{MAX_3D_EXAMPLES})")

                    # --- رسم چک 2 بعدی روی عکس اصلی ---
                    draw_data = metrics['Draw_Data']
                    cv2.drawContours(original_img, [c], -1, (0, 255, 0), 1) 
                    
                    if draw_data['Center_Circle'] and draw_data['Radius'] > 0:
                        cv2.circle(original_img, draw_data['Center_Circle'], draw_data['Radius'], (255, 0, 0), 2)
                    
                    # رسم خط مرکزی 
                    if 'Centerline_Path' in draw_data and len(draw_data['Centerline_Path']) > 1:
                        cv2.polylines(original_img, [draw_data['Centerline_Path']], isClosed=False, color=(0, 0, 255), thickness=2)
                    
                    text_pos = draw_data['Center_Circle'] if draw_data['Center_Circle'] else (c[0][0][0], c[0][0][1])
                    cv2.putText(original_img, f"N{len(contours_found)}", (text_pos[0]-10, text_pos[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    contours_found.sort(key=lambda x: x['Area(um2)'], reverse=True)
    
    for rep_idx, data in enumerate(contours_found):
        row_data = {k: v for k, v in data.items() if k != 'Draw_Data'}
        
        row = {
            'Filename': filename,
            'Replicate': rep_idx + 1
        }
        row.update(row_data)
        flat_data_list.append(row)
        
    cv2.imwrite(os.path.join(CHECK_DIR, f"Checked_{filename}"), original_img)

# ==========================================================
# 5. ایجاد فایل اکسل 
# ==========================================================
print(f"\n[INFO] Saving data to Excel: {EXCEL_OUTPUT}")
try:
    if flat_data_list:
        df = pd.DataFrame(flat_data_list)
        
        with pd.ExcelWriter(EXCEL_OUTPUT, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Nematode_Features', index=False)
            
        print(f"[SUCCESS] Process completed successfully.")
        print(f"[INFO] Advanced Features saved for {len(flat_data_list)} individual nematodes.")
        print(f"Check High-Quality 3D Plots at: {VISUAL_3D_DIR}")
    else:
        print("[WARNING] No nematodes were found in any images.")
        
except Exception as e:
    print(f"[ERROR] Could not save Excel file: {e}")