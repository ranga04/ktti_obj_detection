# KITTI Object Detection and YOLO Conversion

This project utilizes the KITTI dataset to train a YOLOv5 model for object detection. The process involves converting the KITTI dataset annotations to YOLO format, training the YOLOv5 model, and validating the model's performance. This project is intended for research-oriented applications and serves as a robust portfolio piece showcasing proficiency in object detection and dataset handling.

## Project Overview

1. **Dataset**: The [KITTI dataset](http://www.cvlibs.net/datasets/kitti/) provides images and 3D object detection annotations for autonomous driving tasks. This project uses the object detection subset, which includes 7481 annotated training images.
2. **Model**: YOLOv5, a real-time object detection model known for accuracy and speed.
3. **Objective**: Train and validate a YOLO model for object detection using the KITTI dataset, focusing on converting data formats, model training, and bounding box visualization.

## Project Steps

### Step 1: Dataset Preparation and Conversion
   - **Download KITTI Dataset**: Download the object detection dataset from the KITTI website.
   - **Convert Annotations to YOLO Format**: Since the KITTI dataset uses a different format, we need to convert the annotations to YOLO format, which involves:
     - Normalizing bounding box coordinates.
     - Saving annotations as `.txt` files in YOLO format.
   - **Visualize Bounding Boxes**: Verify that the YOLO annotations align correctly by visualizing bounding boxes on sample images.

### Step 2: YOLOv5 Model Training
   - **Clone YOLOv5 Repository**: Clone the YOLOv5 GitHub repository to access training scripts and pretrained weights.
   - **Train the Model**: Use the converted KITTI dataset in YOLO format to train the YOLOv5 model. Training involves setting parameters such as image size, batch size, and epochs.
   - **Validate Model Performance**: Evaluate the model's accuracy on a validation set.

### Step 3: Results Visualization
   - **Visualize Detection Results**: After training, visualize the model's predictions by drawing bounding boxes on images.

## Folder Structure

```
project/
│
├── images/                   # Folder containing images from the KITTI dataset
├── labels/                   # Folder containing YOLO format label files
├── yolov5/                   # YOLOv5 model repository
├── results/                  # Folder to save detection results
└── kitti.yaml                # YOLO dataset configuration file
```

## Dependencies

Install dependencies using the following commands:

```bash
pip install -r yolov5/requirements.txt
pip install tensorflow-datasets opencv-python matplotlib
```

## Usage

1. **Convert KITTI Dataset Annotations**:

   Run the conversion script to convert KITTI annotations to YOLO format:
   ```python
   python convert_kitti_to_yolo.py
   ```

2. **Train the YOLOv5 Model**:

   Navigate to the YOLOv5 directory and run the training script:
   ```bash
   python train.py --img 640 --batch 16 --epochs 10 --data /content/kitti.yaml --weights yolov5s.pt
   ```

3. **Visualize Bounding Boxes**:

   Run the visualization script to verify bounding boxes:
   ```python
   python visualize_boxes.py
   ```

## Sample Code

```python
# Draw bounding boxes on images after training
import cv2
import matplotlib.pyplot as plt

def draw_boxes(image, label_path):
    h, w, _ = image.shape
    with open(label_path, 'r') as f:
        for line in f.readlines():
            _, x_center, y_center, width, height = map(float, line.split())
            x1, y1 = int((x_center - width / 2) * w), int((y_center - height / 2) * h)
            x2, y2 = int((x_center + width / 2) * w), int((y_center + height / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
```

## Results

| Metric       | Value       |
|--------------|-------------|
| Precision    | TBD         |
| Recall       | TBD         |
| mAP@50       | TBD         |
| mAP@50-95    | TBD         |

*Note*: Final model performance is subject to tuning and may vary depending on dataset splits and training parameters.

## Acknowledgments

This project utilizes the KITTI dataset and the YOLOv5 model for object detection. Special thanks to the creators of both resources for making these tools available for research and development.
