# Computer Vision – OpenCV, YuNet & YOLOv8

## Project Overview

This project explores the fundamental and advanced concepts of Computer Vision using Python, OpenCV, YuNet, and YOLOv8.

It combines traditional image processing techniques with Deep Learning-based face and object detection models to understand how computers interpret, process, and analyze visual information.

The project covers image thresholding, morphological transformations, bitwise operations, image histograms, brightness and contrast adjustments, face detection using YuNet, object detection using YOLOv8, and an integrated real-time webcam detection pipeline.

The primary objective is to understand different Computer Vision techniques, compare their functionality, and implement an integrated system capable of detecting faces and everyday objects in real time.

## Project Objectives

- Understand image processing and image manipulation using OpenCV.
- Implement image thresholding and morphological operations.
- Explore erosion, dilation, opening, and closing.
- Perform bitwise image operations using AND, OR, XOR, and NOT.
- Generate grayscale and colour-channel histograms.
- Understand brightness and contrast adjustments.
- Implement face detection using YuNet.
- Perform real-time face detection using a webcam.
- Implement object detection using pretrained YOLOv8n.
- Experiment with confidence and IoU thresholds.
- Analyze object detection results across COCO classes.
- Integrate YuNet and YOLOv8 into a unified detection pipeline.
- Benchmark the FPS performance of individual and combined models.

## Technologies & Libraries

### Programming Language
- Python

### Computer Vision
- OpenCV
- YuNet
- YOLOv8

### Deep Learning
- Ultralytics
- ONNX Model

### Data Processing & Visualization
- NumPy
- Pandas
- Matplotlib

### Development Environment
- Jupyter Notebook
- Webcam / Camera

## Project Workflow

### Task 1: Image Processing & Morphological Operations

This section introduces fundamental image processing techniques using OpenCV.

**1. Image Loading and Binarization**

- Load colour images using OpenCV.
- Convert colour images into grayscale.
- Apply binary thresholding with a threshold value of 127.
- Compare grayscale and binary images.

**2. Morphological Operations**

Implement four fundamental morphological operations using a 5×5 rectangular kernel:

- **Erosion:** Shrinks foreground objects and removes small bright noise.
- **Dilation:** Expands foreground objects and fills small gaps.
- **Opening:** Applies erosion followed by dilation to remove small foreground noise.
- **Closing:** Applies dilation followed by erosion to fill small holes and gaps.

**3. Kernel Shape and Size Comparison**

Compare morphological operations using three different kernel shapes:

- RECT
- ELLIPSE
- CROSS

Experiment with kernel sizes:

- 3×3
- 5×5
- 9×9

Analyze how different kernel shapes and sizes affect image structure and noise removal.

### Task 2: Bitwise Operations & Image Histograms

This section explores logical image operations and intensity distribution analysis.

**1. Bitwise Operations**

Implement the following operations using OpenCV:

- Bitwise AND
- Bitwise OR
- Bitwise XOR
- Bitwise NOT

Create circular and rectangular masks and visualize their individual and combined results.

**2. Masked Image Overlay**

- Extract a circular Region of Interest (ROI) from an image.
- Apply a binary mask to isolate a selected image region.
- Demonstrate a simple logo overlay using bitwise OR.

**3. Image Histograms**

Generate and visualize:

- Grayscale intensity histogram.
- Blue channel histogram.
- Green channel histogram.
- Red channel histogram.

Understand how pixel intensities are distributed across different colour channels.

**4. Brightness and Contrast Adjustment**

Apply brightness and contrast transformations using:

`new_pixel = alpha × pixel + beta`

Experiment with:

- Brighter image: alpha = 1.5, beta = 30.
- Darker image: alpha = 0.7, beta = -30.

Compare the original, brighter, and darker images along with their corresponding grayscale histograms.

### Task 3: Face Detection Using YuNet

Implement face detection using the YuNet ONNX model through OpenCV's FaceDetectorYN API.

**1. Model Setup**

- Download the pretrained YuNet ONNX model.
- Check OpenCV compatibility.
- Initialize the YuNet face detector.

**2. Static Image Face Detection**

Perform face detection on sample images.

Visualize:

- Face bounding boxes.
- Five facial landmarks.
- Detection confidence scores.

**3. Real-Time Webcam Detection**

Implement a webcam-based face detection function.

Features include:

- Continuous frame processing.
- Real-time face detection.
- Facial landmark visualization.
- FPS calculation.
- Webcam exit using the `q` key.

**4. Score Threshold Experiment**

Compare face detection results using three confidence thresholds:

- 0.5
- 0.7
- 0.9

Observe how different thresholds affect the number of detected faces.

**5. Privacy Protection Through Face Blurring**

Apply Gaussian Blur to detected face regions to demonstrate a basic privacy-preserving image processing application.

### Task 4: Object Detection Using YOLOv8

This section implements object detection using the pretrained YOLOv8 Nano model.

**1. Model Initialization**

- Load the pretrained YOLOv8n model.
- Use pretrained COCO weights.
- Perform object detection on static images.
- Visualize bounding boxes, class labels, and detection results.

**2. Real-Time Webcam Object Detection**

Implement real-time object detection through a webcam.

Features include:

- Continuous video frame capture.
- Object classification.
- Bounding box visualization.
- Confidence-based detection filtering.
- FPS calculation.

**3. Confidence and IoU Threshold Tuning**

Compare different confidence thresholds:

- 0.25
- 0.50
- 0.75

Experiment with IoU thresholds:

- 0.3
- 0.5
- 0.7

Analyze how confidence filtering and Intersection over Union (IoU) thresholds influence retained detection boxes and Non-Maximum Suppression (NMS).

**4. Per-Class Detection Summary**

- Extract predicted object classes.
- Count detections across the sample images.
- Visualize detection counts for different COCO classes using a horizontal bar chart.

### Task 5: Integrated Computer Vision Pipeline

This section combines classical image processing and Deep Learning-based detection techniques.

**1. Unified Detection Pipeline**

Integrate:

- YuNet for face detection.
- YOLOv8n for general object detection.

The combined webcam pipeline displays:

- Green bounding boxes for detected faces.
- Face confidence scores.
- YOLO object bounding boxes.
- Object class labels.
- Real-time FPS information.

**2. Morphological Pre-Cleaning**

Explore the effect of morphological preprocessing on webcam detection.

Steps include:

- Convert frames to grayscale.
- Apply binary thresholding.
- Perform morphological opening.
- Compare detection performance with and without preprocessing.

The preprocessing option is experimental because threshold masking can remove useful visual information.

**3. FPS Benchmarking**

Benchmark three configurations using the same captured webcam frames:

- YuNet-only face detection.
- YOLOv8-only object detection.
- Combined YuNet and YOLOv8 inference.

The benchmark captures up to 100 frames and calculates FPS for each configuration.

Actual performance depends on hardware, image resolution, and available computational resources.

**4. Final Results Comparison**

Compare the techniques implemented throughout the project:

| Technique | Category | Primary Application |
|---|---|---|
| Morphological Operations | Classical CV | Binary mask cleaning |
| Bitwise Operations | Classical CV | Masking and ROI extraction |
| Histograms | Classical CV | Intensity analysis |
| YuNet | Deep Learning | Face detection |
| YOLOv8n | Deep Learning | General object detection |
| Integrated Pipeline | Hybrid CV | Combined face and object detection |

FPS values should be measured on the actual system rather than estimated.

## Key Computer Vision Concepts Covered

- Image Processing
- Image Thresholding
- Binary Images
- Morphological Transformations
- Structuring Elements and Kernels
- Bitwise Image Operations
- Region of Interest (ROI)
- Image Histograms
- Brightness and Contrast
- Face Detection
- Facial Landmark Detection
- Object Detection
- Bounding Boxes
- Confidence Thresholds
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS)
- Real-Time Video Processing
- Frames Per Second (FPS)
- Deep Learning-Based Computer Vision

## Project Structure

```text
Computer-Vision-OpenCV-YuNet-YOLOv8/
│
├── CV_PR3.ipynb
│
├── data/
│   ├── images/
│   │   ├── baboon.jpg
│   │   ├── bus.jpg
│   │   ├── zidane.jpg
│   │   └── smarties.png
│   │
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
│
├── plots/
│   ├── morphology_grid.png
│   ├── opening_kernel_grid.png
│   ├── closing_kernel_grid.png
│   ├── bitwise_grid.png
│   ├── masked_overlay.png
│   ├── grayscale_histogram.png
│   ├── colour_histograms.png
│   ├── histogram_comparison.png
│   └── yolo_detections.png
│
└── README.md
```

**Note:** Images and the YuNet model are downloaded automatically by the notebook where possible. The YOLOv8n weights (`yolov8n.pt`) are downloaded by Ultralytics when the model is initialized. Generated plots are saved in the `plots/` directory.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Computer-Vision-OpenCV-YuNet-YOLOv8.git
```

### 2. Navigate to the Project Directory

```bash
cd Computer-Vision-OpenCV-YuNet-YOLOv8
```

### 3. Install Required Dependencies

```bash
pip install opencv-contrib-python ultralytics matplotlib numpy pandas jupyter
```

**Important:** The notebook specifies OpenCV 5.0.0.* packages. If that version is unavailable for your Python environment, install a compatible OpenCV build that supports `FaceDetectorYN`.

Avoid installing conflicting `opencv-python` and `opencv-contrib-python` distributions together in the same environment.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Execute the Notebook

Open `CV_PR3.ipynb` and run the cells sequentially.

Ensure the required dependencies and model files are available.

## Running the Webcam Detection Functions

The notebook includes three optional webcam functions.

**YuNet Face Detection**

```python
run_yunet_webcam()
```

**YOLOv8 Object Detection**

```python
run_yolo_webcam()
```

**Integrated Face and Object Detection**

```python
run_integrated_pipeline()
```

**FPS Benchmarking**

```python
benchmark_pipeline_fps()
```

Press `q` to exit the webcam windows.

Note: Webcam functionality requires a local environment with camera access. Cloud notebook environments may not support direct webcam access.

## Applications

The techniques implemented in this project have practical applications in:

- Smart surveillance systems.
- Real-time face detection.
- Privacy protection through face blurring.
- Automated object recognition.
- Traffic and vehicle monitoring.
- Image preprocessing and enhancement.
- Smart camera systems.
- Edge-based computer vision applications.
- Real-time visual monitoring.

## Key Learnings

This project provides practical experience in:

- Understanding traditional Computer Vision techniques.
- Performing image transformations using OpenCV.
- Working with binary masks and morphological operations.
- Analyzing image intensity and colour distributions.
- Implementing pretrained face detection models.
- Using ONNX models with OpenCV.
- Working with YOLOv8 for object detection.
- Understanding detection thresholds and NMS.
- Integrating multiple detection models.
- Evaluating real-time performance using FPS benchmarking.

## Conclusion

This project demonstrates the practical implementation of Computer Vision through a combination of traditional image processing and modern Deep Learning techniques.

By exploring morphological operations, bitwise transformations, histograms, YuNet face detection, and YOLOv8 object detection, the project develops an understanding of how different computer vision methods work individually and together.

The final integrated pipeline combines face and general object detection in a real-time webcam environment, demonstrating the potential of hybrid computer vision systems for practical applications.

The project also highlights the importance of preprocessing, threshold selection, computational efficiency, and performance benchmarking when developing real-time detection systems.

---

**Project Domain:** Computer Vision | Deep Learning | Image Processing | Object Detection

**Technologies:** Python | OpenCV | YuNet | YOLOv8 | TensorFlow-independent Ultralytics inference | NumPy | Pandas | Matplotlib
