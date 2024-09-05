# Object Detection Project

## Overview

This project utilizes a pre-trained Faster R-CNN model from the `torchvision` library to perform object detection. You can either perform live object detection using your webcam or process a single image file.

## Getting Started

### Prerequisites

- Python 3.9
- create conda environment in your local machine

### Creating Environment in your machine

- Open anaconda prompt
- conda create -n your_environment_name python=3.9
- conda activate your_environment_name
- download all required libraries/modules from requirements.txt file
- Install all required dependencies from requirements.txt file

### Cloning the Repository
   
- Link: [https://github.com/mognc/object-detection.git](https://github.com/mognc/Object-Detection)
- Guide: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

### Running the Project

- Navigate to directory in which you cloned the repository
- From command prompt cd your_repo_directory
- conda activate your_environment

#### Live Object Detection

- python main.py --mode live

#### Image File Detection

- python main.py --mode image --image_path path/to/your/image.jpg

### Arguments

--mode: Specifies the mode of operation. Acceptable values are:
- 'live': For live object detection using your webcam.
- 'image': For processing a single image file.
  
--image_path: Required if --mode is set to 'image'. The path to the image file you want to process.
   
