# YOLOv1 Object Detection on PASCAL VOC

<img width="621" alt="image" src="https://github.com/user-attachments/assets/4fc0bafc-def0-4e30-9d10-0b475237c4ea" />

> A PyTorch implementation of the YOLO (You Only Look Once) object detection algorithm trained on the PASCAL VOC 2007 dataset, achieving a minimum mAP of 0.5 after 50 epochs

## Project Overview

This project implements the YOLO (You Only Look Once) object detection algorithm applied to the PASCAL VOC 2007 dataset. YOLO is a state-of-the-art real-time object detection system that processes images in a single network evaluation, making it significantly faster than previous detection methods while maintaining competitive accuracy.

Unlike traditional object detection approaches that rely on region proposal and subsequent classification, YOLO frames detection as a regression problem to spatially separated bounding boxes and associated class probabilities. This unified approach allows for end-to-end training and real-time performance.

## Architecture & Implementation Details

### Core Components:

1. **Network Backbone**: ResNet50 architecture pretrained on ImageNet, adapted for the detection task
2. **Output Grid**: 14×14 grid, with each cell predicting 2 bounding boxes (differs from original YOLO's 7×7 grid)
3. **Loss Function**: Custom YOLO loss combining localization error, confidence error, and classification error
4. **Bounding Box Prediction**: Each cell predicts B=2 boxes, each with 5 parameters (x, y, w, h, confidence)
5. **Class Prediction**: Each grid cell predicts class probabilities for 20 PASCAL VOC categories

### Technical Specifications:

- **Model Design**: ResNet50 backbone with custom detection head
- **Loss Components**: 
  - Bounding box coordinate loss (with higher λ_coord weight for better localization)
  - Object confidence loss
  - No-object confidence loss (with lower λ_noobj weight to address class imbalance)
  - Classification loss
- **Hyperparameters**:
  - Learning Rate: 0.001 with scheduled reductions
  - Batch Size: 16
  - λ_coord = 5
  - λ_noobj = 0.5

## Features & Results

### Detection Performance

- Achieved a minimum mAP (mean Average Precision) of 0.5 on the PASCAL VOC test set
- Training progression:
  - Epoch 5: mAP = 0.0296
  - Epoch 10: mAP = 0.2092
  - Epoch 15: mAP = 0.3366
  - Epoch 20: mAP = 0.4113
  - Epoch 25: mAP = 0.4556
  - Epoch 30: mAP = 0.4857

### Visualization and Debugging

- Built-in visualization of object detection results
- Performance evaluation using mAP (mean Average Precision)
- Support for displaying bounding boxes with class names and confidence scores

## Implementation Pipeline

1. **Data Preparation**: Loading and processing PASCAL VOC 2007 dataset
2. **Network Initialization**: Loading pretrained ResNet50 backbone
3. **YOLO Loss Implementation**: Custom implementation of YOLO's loss function
4. **Training Loop**: Training the network with SGD optimizer for 50+ epochs
5. **Evaluation**: Computing mAP on the test set
6. **Visualization**: Displaying detection results on test images
7. **Kaggle Submission**: Generating output for the Kaggle competition

## Getting Started

### Dependencies

```
torch
torchvision
numpy
matplotlib
opencv-python (cv2)
```

### Dataset Preparation

The PASCAL VOC 2007 dataset is automatically downloaded by the provided script:

```bash
bash ./download_data.sh VOC_PATH
```

### Training the Model

The notebook guides you through training the model:

```python
# Initialize the network
net = resnet50(pretrained=True).to(device)

# Set up loss function and optimizer
criterion = YoloLoss(S=14, B=2, lambda_coord=5, lambda_noobj=0.5)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Training loop
for epoch in range(num_epochs):
    # Train for one epoch
    # Evaluate model
    # Save checkpoints
```

### Evaluating the Model

```python
# Evaluate on the test set
test_aps = evaluate(net, test_dataset_file=annotation_file_test, img_root=file_root_test)
```

## Loss Function Implementation

The core of the YOLO algorithm is its multi-part loss function, which handles:

1. **Bounding box coordinate regression**: Penalizes errors in the predicted box coordinates, focusing on the square root of width and height to better handle differences between large and small boxes
2. **Object confidence scoring**: Measures how confident the model is that a box contains an object and how accurate the box is
3. **No-object confidence suppression**: Penalizes boxes in cells where no objects exist
4. **Class probability prediction**: Classifies what type of object is contained in each grid cell

```python
class YoloLoss(nn.Module):
    def __init__(self, S, B, lambda_coord, lambda_noobj):
        # Initialize YOLO loss
        
    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        # Compute loss components
        # Return total loss and individual loss components
```

## Background Theory

This implementation follows the architecture described in the [original YOLO paper](https://pjreddie.com/media/files/papers/yolo_1.pdf) by Redmon et al. The key innovation of YOLO is framing object detection as a regression problem, where a single network predicts bounding boxes and class probabilities directly from full images in one evaluation.

The workflow consists of:
1. Dividing the input image into an S×S grid
2. Each grid cell predicts B bounding boxes with confidence scores
3. Each grid cell also predicts C conditional class probabilities
4. The final detection combines these predictions

## Acknowledgments

This project was completed as part of CS 444 Assignment 4. The implementation is based on the original YOLO paper by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.

## Additional Resources

- [Original YOLO Paper](https://pjreddie.com/media/files/papers/yolo_1.pdf)
- [YOLO Loss Function Explanation](https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
