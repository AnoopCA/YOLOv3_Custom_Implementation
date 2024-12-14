# YOLOv3 Custom Implementation

## Description
This project implements an object detection system using the YOLO (You Only Look Once) algorithm. The system is designed to detect objects in images efficiently and accurately, making it suitable for real-time applications. It leverages the Pascal VOC dataset and incorporates advanced data augmentation and loss calculation techniques to optimize performance. Debugged and optimized the model parameters to ensure compatibility with a system of average specifications.

## Features
- **YOLO Algorithm**: Implementation of the YOLO architecture for multi-scale object detection.
- **Data Augmentation**: Advanced image transformations for robust model training using Albumentations.
- **Loss Function**: Custom YOLO loss function to handle objectness, bounding boxes, and class predictions.
- **Support for Pascal VOC Dataset**: Ready-to-use integration with the Pascal VOC dataset.
- **Anchor Boxes**: Optimized anchor box handling for better localization.

## Technologies Used
- **Programming Languages**: Python
- **Libraries and Frameworks**: 
  - PyTorch
  - Albumentations
  - NumPy
  - pandas
  - OpenCV
  - PIL (Python Imaging Library)

## Data
- **Dataset**: Pascal VOC
  - **Images**: Various categories like aeroplane, bicycle, bird, car, etc.
  - **Annotations**: Bounding box labels in YOLO format.
- **Source**: Pascal VOC dataset [link](http://host.robots.ox.ac.uk/pascal/VOC/).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnoopCA/YOLOv3_Custom_Implementation.git
   cd yolo-object-detection

## Inspiration
The custom implementation of YOLOv3 is originally written by Aladdin Persson, based on the YOLOv3 research paper. You can find the original implementation and more of his work on his [GitHub profile](https://github.com/aladdinpersson). The code has been cloned locally and pushed to this repository for experimental purposes.

## About YOLOv3
YOLO (You Only Look Once) is a real-time object detection system. YOLOv3 is the third version of the YOLO architecture, renowned for its speed and accuracy in object detection tasks.
