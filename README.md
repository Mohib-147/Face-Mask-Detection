# Face-Mask-Detection

# Mask Detection System

## 1- Introduction

**Motivation:** Masks have become an important part of daily life in hospitals, public spaces, and high-risk environments. But just wearing a mask is not always enough, it has to be worn correctly to actually work. A system that can automatically check whether a person is wearing a mask, wearing it properly, and what type of mask it is would be very useful in real-world settings.

**Problem Statement:** Build a computer vision system that detects whether a person is wearing a mask or not, checks if the mask is worn properly (fully covering nose and mouth or just partially), identifies the type of mask (surgical, N95, cloth, etc.), and also detects the color of the mask.

## 2- Background/Literature Review

Face mask detection has been studied a lot, especially after COVID-19. Early systems used basic image processing methods like edge detection and color filtering to spot masks. Later, machine learning models like Support Vector Machines were trained on facial features to classify mask vs. no mask. More recently, deep learning models like CNNs (Convolutional Neural Networks) and YOLO (You Only Look Once) have become the standard because they are fast, accurate, and can detect multiple things in a single image at the same time. Our system builds on this by going beyond simple detection — we aim to also classify mask type and fit quality.

## 3- Dataset Details

We will use a combination of publicly available datasets and physically recorded images/videos. The data will include:

- People with no mask
- People with a mask worn properly (nose and mouth fully covered)
- People with a mask worn improperly (only mouth covered, mask pulled down, etc.)
- Different mask types: surgical masks, N95/KN95 masks, cloth masks, face shields
- Different mask colors: white, black, blue, patterned, etc.
- Different lighting conditions, angles, and backgrounds to make the model more robust

We will manually label all images with bounding boxes and class labels. Bad quality images will be removed, and the dataset will be split into training, validation, and test sets.

## 4- Proposed Methodology

### Pipeline

```
Raw Image / Video Frame
        ↓
Preprocessing
resize, normalize pixel values, augment data
        ↓
Face Detection
locate faces in the image using a face detector
        ↓
Mask Detection Model
CNN / YOLOv8 — classifies each detected face into:
No Mask / Mask Worn Properly / Mask Worn Improperly
        ↓
Mask Type Classifier
identifies mask type: Surgical / N95 / Cloth / Shield
        ↓
Color Detection
identifies dominant mask color using HSV color space analysis
        ↓
Output
Bounding box around face plus mask status, mask type, mask color
        ↓
Post Processing (Optional)
log detections, flag no-mask or improper-mask events, display summary stats
```

### Model Architecture

We plan to use YOLOv8 as our main model because it handles object detection and classification together in a single pass, making it fast and accurate. We will train it on our labeled dataset with multiple output classes: No Mask, Proper Mask, Improper Mask. A separate lightweight CNN will handle mask type and color classification after the face region is cropped.

### Training Details

- **Loss Function:** Cross Entropy Loss for classification tasks
- **Optimizer:** Adam
- **Data Augmentation:** flipping, rotation, brightness changes to improve generalization
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, mean Average Precision
