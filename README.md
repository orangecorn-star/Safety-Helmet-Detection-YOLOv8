# Safety Helmet Detection in Long-Tail Scenarios Based on YOLOv8 and Hard Sample Mining
This project addresses the Helmet missed inspection problem caused by long-tail conditions such as rain, snow, low illumination, and occlusion in industrial scenarios. Based on YOLOv8, we build a complete object detection pipeline and design an ablation study featuring "dual-channel data fusion" and "hard sample isolated testing".

# 🛠️ 1. Environment Setup
Ensure that Anaconda or Miniconda is installed on your system, along with an NVIDIA GPU that supports CUDA (this experiment is tuned on RTX 4050 6GB).

```bash
# 1. Clone this repository
git clone https://github.com/orangecorn-star/Safety-Helmet-Detection-YOLOv8.git
cd Safety-Helmet-Detection-YOLOv8

# 2. Create and activate the virtual environment
conda create -n yolov8_env python=3.10
conda activate yolov8_env

# 3. Install dependencies (PyTorch and Ultralytics)
pip install -r requirements.txt
```

# 📂 2. Data Preparation
This project uses two data sources: a public base dataset and a self-constructed hard sample set.

Base Public Dataset (5000+ images): Due to its large size, it is not included in this repository. Please download it from the open-source community and place it under the datasets/images/ directory.

Self-Constructed Hard Sample Set (200 images): Fully open-sourced in this repository! It covers long-tail scenarios including severe occlusion, rain/fog, and nighttime low illumination. Unzip datasets/hard_samples_200.zip to obtain the original images and manually annotated labels.

To reproduce the full experiment, unzip the hard samples and run the following preprocessing scripts:
```bash
# 1. Convert base dataset from VOC to YOLO format with coordinate normalization
python tools/prepare_data.py

# 2. Inject 200 manually annotated hard samples into the training set
python tools/merge_hard_samples.py

# 3. (Optional) Visualize the size distribution differences between the two datasets
python tools/plot_data_distribution.py
```

# 🚀 3. Training
This project supports two comparison modes: Baseline and Enhanced (with hard samples). To resolve process deadlock issues on Windows, all training scripts have workers=0 set.
```bash
# Train YOLOv8 baseline model (base data only)
python train_yolov8.py

# Train YOLOv8 enhanced model (with long-tail hard samples)
python train_yolov8_hard.py

# (Optional) Train YOLOv10 for cross-generation architecture comparison
python train_yolov10.py
```

# 📊 4. Evaluation & Visualization
To rigorously quantify the benefits of the data augmentation strategy, 40 hard samples are isolated from the validation set for independent blind testing.

Quantitative Evaluation:
```bash
# Run hard-sample isolated validation (computes Recall and mAP gains)
python val_hard_only.py

# Plot Loss and mAP comparison curves between Baseline and Enhanced models
python plot_curves.py
```

Qualitative Evaluation (Inference Testing):
```bash
# Run side-by-side comparison, generating red/green bounding box results
python compare_test1.py

# Run real-time video stream inference
python VideoTest.py
```

# 🧩 5. Project Structure
```text
Safety-Helmet-Detection-YOLOv8/
│
├── datasets/                      # Dataset directory
│   ├── hard_samples_200.zip       # Self-constructed hard sample set (compressed)
│   ├── data.yaml                  # Base dataset configuration
│   └── hard_data.yaml             # Configuration after hard sample injection
│
├── weights/                       # Pretrained weights
│   ├── yolov8n.pt                 # YOLOv8n official pretrained weights
│   └── yolov10n.pt                # YOLOv10n official pretrained weights
│
├── tools/                         # Data processing and analysis tools
│   ├── prepare_data.py            # VOC to YOLO format conversion & dataset split
│   ├── merge_hard_samples.py      # Hard sample fusion script (8:2 split)
│   ├── analyze_dataset.py         # Bounding box size distribution statistics
│   └── plot_data_distribution.py  # Dataset distribution histogram plotting
│
├── utils/                         # Utility and configuration modules
│   ├── Config.py                  # Global path and hyperparameter configuration
│   └── detect_tools.py            # Drawing, logging, and other low-level utilities
├── requirements.txt               # Python dependencies
├── README.md                      # Core documentation
├── .gitignore                     # Git ignore rules (prevents uploading large runs/ and datasets/)
│
# --- Core scripts (located in root directory) ---
├── train_yolov8.py                # Baseline model training
├── train_yolov8_hard.py           # Enhanced model training (with hard samples)
├── train_yolov10.py               # Cross-generation comparison training (YOLOv10)
├── val_hard_only.py               # Hard-sample isolated validation & evaluation
├── plot_curves.py                 # Training curve comparison plotting
├── compare_test1.py               # Qualitative evaluation: red/green comparison results
├── imgTest.py                     # Single image inference test
└── VideoTest.py                   # Real-time video stream inference test
```

# 📄 6. License
This project is for educational purposes only. The self-constructed hard sample dataset is collected from public internet resources for non-commercial research use. All copyrights belong to their respective owners.