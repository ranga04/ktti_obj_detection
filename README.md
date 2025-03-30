# KITTI-to-YOLO Conversion and YOLOv5 Setup

This repository contains a Jupyter Notebook for converting the KITTI dataset for object detection into the YOLO format and setting up the YOLOv5 framework for training and inference. Due to limited computational resources, the model training has not been executed, but all conversion and setup scripts are provided so that the project can be easily extended or executed on a more powerful machine.

## Overview

The KITTI dataset is a benchmark for autonomous driving research. This project demonstrates how to:
- Download and prepare the KITTI dataset using TensorFlow Datasets.
- Convert KITTI annotations (bounding boxes) into YOLO format.
- Organize the dataset into training, validation, and test splits.
- Set up the YOLOv5 repository with a custom data configuration for KITTI.
- (Optionally) Train the YOLOv5 model and run inference on test images.

> **Note:** Training has not been performed due to limited computational resources. The provided notebook and scripts perform data conversion and setup, allowing for future training when resources are available.

## Directory Structure

After running the provided notebook, the repository structure will look similar to:

```
├── notebooks/                   # Jupyter notebooks for experiments and visualization
│   └── kitti.ipynb              # Main notebook for dataset conversion and visualization
├── yolo_kitti/                  # Converted KITTI dataset in YOLO format
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── yolov5/                      # Cloned YOLOv5 repository (see setup instructions below)
│   ├── data/
│   │   └── kitti.yaml           # Data configuration file for KITTI
│   ├── train.py                 # Training script (not executed by default)
│   └── detect.py                # Inference script
└── README.md                    # This file
```

## Requirements

- **Python:** 3.9 or later (recommended)
- **Conda:** (Optional but recommended for managing environments)
- **Libraries:** TensorFlow, TensorFlow Datasets, Matplotlib, scikit-learn, OpenCV, PyTorch, and additional dependencies for YOLOv5

## Installation

### 1. Create a Conda Environment

```bash
conda create -n kitti_project39 python=3.9
conda activate kitti_project39
```

### 2. Install Required Python Packages

```bash
conda install -c conda-forge jupyter notebook tensorflow tensorflow-datasets matplotlib scikit-learn opencv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Clone the YOLOv5 Repository and Install Dependencies

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## Data Preparation

1. **Run the Notebook:**
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Navigate to the `notebooks` folder and open `kitti.ipynb`.
   - The notebook downloads and prepares the KITTI dataset, visualizes sample images with bounding boxes, and converts the annotations into the YOLO format.
   - The converted dataset will be organized under the `yolo_kitti` folder with subdirectories for images and labels (split into train, val, and test).

2. **Verify Conversion:**
   - Check that the `yolo_kitti` directory contains images and corresponding label files in YOLO format (each label file contains lines like: `<class_id> <x_center> <y_center> <width> <height>`, with values normalized to the image dimensions).

## YOLOv5 Setup

### Create the Data Configuration File

In the `yolov5/data` folder, create a file named `kitti.yaml` with the following content:

```yaml
train: ../yolo_kitti/images/train  # Path to training images
val: ../yolo_kitti/images/val      # Path to validation images
nc: 8                              # Number of classes
names: ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
```

### (Optional) Training

If you have access to a machine with sufficient computational resources, you can train the model using:

```bash
python train.py --img 640 --batch 16 --epochs 10 --data data/kitti.yaml --weights yolov5s.pt --project kitti_training --name yolov5s_kitti
```

*Parameters:*
- `--img 640`: Input image size.
- `--batch 16`: Batch size (adjust if necessary).
- `--epochs 10`: Number of epochs.
- `--data data/kitti.yaml`: Path to the data configuration file.
- `--weights yolov5s.pt`: Pre-trained weights.
- `--project kitti_training --name yolov5s_kitti`: Output directory for logs and checkpoints.

## Inference

After training (or using pre-trained weights), run inference on test images with:

```bash
python detect.py --weights kitti_training/yolov5s_kitti/weights/best.pt --img 640 --conf 0.25 --source path/to/test_images
```

Replace `path/to/test_images` with the path to a single image, a directory of images, or a video file.

## Future Work

- **Evaluation:** Fine-tune hyperparameters, experiment with data augmentation, and compare performance with other object detection models.
- **Extensions:** Explore additional tasks (e.g., 3D object detection, multi-task learning) using the KITTI dataset.

## Acknowledgements

- **KITTI Dataset:** [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- **YOLOv5:** Ultralytics – [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- **TensorFlow Datasets:** [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets)

## License

This project is licensed under the MIT License. 

