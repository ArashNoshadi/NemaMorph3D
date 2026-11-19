# NemaMorph3D

**Deep learning instance segmentation and 3D biomorphometric phenotyping of root-knot nematodes (*Meloidogyne javanica*) in eggplant roots**

NemaMorph3D is the code accompanying the manuscript *"Deep learning-enabled instance segmentation and 3D biomorphometric phenotyping reveal spatiotemporal dynamics of root-knot nematode infection in eggplant"* (submitted to *Scientific Reports*).

The pipeline takes brightfield micrographs of acid fuchsin-stained roots, segments intra-radical nematodes with a YOLO11 instance-segmentation model, and converts every segmentation mask into a set of 2D and 3D morphometric descriptors (length, width, volume, sphericity, surface-to-volume ratio, texture, etc.).

---

## Contents

| File | Purpose |
|---|---|
| `Train_YOLOv11-seg.py` | Builds the YOLO dataset from raw images and masks (CLAHE, polygon labels, 80/10/10 split), trains YOLO11-seg, and runs a first hybrid evaluation (detection + texture filter) on the test split. |
| `yolo_evaluator.py` | Full post-training evaluation for the paper: standard YOLO validation (mAP, PR curves), image-level ROC/AUC ("has nematode"), count regression metrics with bootstrap CIs, Bland-Altman, calibration, robustness to noise, scale sensitivity, latency, model size, and publication-ready figures and Excel tables. |
| `mesurment_v4.py` | Runs the trained model on new images and extracts per-nematode morphometrics, including the 3D height-map reconstruction and optional 1200-dpi 3D renders. Writes `Nematode_Morphology_Report.xlsx`. |

## Workflow

```
raw micrographs + binary masks
          │
          ▼
Train_YOLOv11-seg.py  ──►  best.pt  +  data.yaml  +  test report
          │
          ├──►  yolo_evaluator.py   ──►  metrics, figures, Excel tables
          │
          ▼
mesurment_v4.py       ──►  per-nematode 2D/3D features (.xlsx) + 3D renders
```

## Requirements

- Python 3.9 or newer
- A CUDA-capable GPU is strongly recommended. The scripts were developed on an NVIDIA RTX 3080 Ti (12 GB).
- Python packages:

```bash
pip install ultralytics torch opencv-python numpy pandas scipy scikit-learn \
            matplotlib seaborn openpyxl pyyaml thop
```

## Data layout

`Train_YOLOv11-seg.py` expects two folders (edit `CONFIG['raw_dir']` and `CONFIG['mask_dir']`):

- **Raw images** (`.jpg`, `.png`, `.bmp`, `.tif`, `.tiff`).
- **Binary masks** with one of these names per image: `contour_<name>.png/.jpg/.tif` or `<name>.png/.tif`.
- Images **without** a matching mask are treated as **hard negatives** (background-only frames with empty labels). They are kept in the dataset to reduce false positives from optical flare, bubbles and root debris.

Images are converted to grayscale, resized to `img_size`, enhanced with CLAHE, and masks are converted to simplified YOLO polygons (single class: `nematode`).

## Usage

### 1. Train

Edit the paths and hyperparameters in `CONFIG` at the top of `Train_YOLOv11-seg.py`, then run:

```bash
python Train_YOLOv11-seg.py
```

All outputs (log, weights, visual checks, Excel report) are written to a time-stamped `Run_Master_<timestamp>` folder.

Main settings in `CONFIG` (script defaults):

| Parameter | Default | Note |
|---|---|---|
| `model_type` | `yolo11l-seg.pt` | Any Ultralytics YOLO11-seg checkpoint, e.g. `yolo11n-seg.pt`. |
| `img_size` | 1024 | Square resize applied to images and masks. |
| `batch_size` | 8 | Reduce for smaller GPUs. |
| `epochs` / `patience` | 200 / 40 | Early stopping on validation. |
| `noise_prob` | 0.3 | Probability of adding a Gaussian-noise copy of a training image. |
| `min_area` | 50 | Minimum contour area (px) kept as a label or prediction. |
| `texture_variance_threshold` | 150.0 | Laplacian-variance cut-off used in the hybrid evaluation to reject smooth artifacts (bubbles, threads). |

Training uses AdamW (lr0 = 0.001, cosine schedule), mixed precision, mosaic augmentation, flips, rotation, translation and scaling.

### 2. Evaluate

`yolo_evaluator.py` locates the latest `Run_Master_*` folder automatically and writes everything to `Paper_Results_Full/`:

```bash
python yolo_evaluator.py
```

It produces `Results_Data_Tables.xlsx`, `Paper_Metrics_Report.txt`, and PNG/PDF figures (loss and mAP curves, ROC and PR curves, confusion matrix, parity and Bland-Altman plots, calibration, robustness, latency, etc.). The ROC curve is computed at **image level** (nematode present or absent).

### 3. Measure nematodes and build 3D reconstructions

Set `RAW_IMAGE_DIR`, `MODEL_PATH`, `BASE_OUTPUT_DIR` and the **spatial calibration** in `mesurment_v4.py`, then run:

```bash
python mesurment_v4.py
```

> **Calibration:** `KNOWN_PIXELS` / `KNOWN_MICRONS` (default 1030.855 px = 10 µm) must be replaced with the calibration of your microscope, objective and camera. All physical units depend on it.

Outputs: `Nematode_Morphology_Report.xlsx` (one row per nematode), 2D check overlays, and up to `MAX_3D_EXAMPLES` randomly selected 3D renders.

## Extracted features

| Feature | Definition in the code |
|---|---|
| `Length(um)` | Geodesic centerline length (shortest path weighted toward the body axis, smoothed with a cubic spline). Falls back to the longest hull diameter. |
| `Max_Width(um)` | 2 × maximum of the Euclidean distance transform of the mask. |
| `Shape_Class` | `Elongated` if length / width ≥ 3, otherwise `Round/Cyst`. |
| `Area(um2)`, `Perimeter(um)` | From the mask contour. |
| `Volume_3D(um3)` | Sum of 2·Z over the mask, where Z is a height map built from the distance transform and the skeleton (hemispherical cross-section), then morphologically closed and Gaussian-smoothed. |
| `Surface_to_Volume_Ratio` | Approximate surface area ÷ 3D volume, with surface = 2·area + perimeter · 2·mean(Z). |
| `Sphericity_Index` | π^(1/3) · (6V)^(2/3) / surface area. |
| `Globosity_Index` | Width ÷ length. |
| `Tortuosity` | (Perimeter / 2) ÷ length. |
| `Cytoplasm_Texture(std)` | Standard deviation of grayscale intensity inside the mask. |
| `Thickness_Heterogeneity` | Standard deviation of the distance transform inside the mask. |
| `Mass_Shift_Index` | Distance between centroid and bounding-box centre, normalized by length. |
| `Allometric_Area` | Area ÷ length². |
| `Weight_Estimate(ug)` | Volume × density of 1.13. |

## Reproducibility notes

- The train/validation/test split uses `random_state = 42`.
- Paths in the three scripts are hard-coded (Linux paths in the training and evaluation scripts, Windows paths in the measurement script). Edit them before running.
- The script defaults above may differ from the settings reported in the manuscript. Set `model_type`, `img_size`, `batch_size`, `epochs` and `patience` to the values used in the paper before reproducing the results.

## Data and model availability

Trained weights, annotated images and masks, and the extracted feature tables are described in the Data and Code Availability sections of the manuscript. A link will be added here upon publication.

## Citation

If you use this code, please cite:

> Noshadi A., Khaji F., Chadegani H. *Deep learning-enabled instance segmentation and 3D biomorphometric phenotyping reveal spatiotemporal dynamics of root-knot nematode infection in eggplant.* 

## License

To be specified (for example MIT or Apache-2.0). Add a `LICENSE` file to the repository.

## Contact

Nowshadiarash@gmail.com

