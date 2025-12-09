# CNNs Based Deepfake Detection & Face Anti-Spoofing

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Research_Project-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/University-HCMUTE-red)](http://hcmute.edu.vn/)

> **A comprehensive research project on detecting biometric spoofing attacks (Deepfakes, Replay, Print Attacks) using custom and State-of-the-Art Convolutional Neural Networks.**

---

## 📋 Table of Contents
- [Abstract](#-abstract)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Dataset Preparation](#-dataset-preparation)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#1-training)
  - [Evaluation](#2-evaluation)
  - [Inference](#3-inference)
- [Model Architectures](#-model-architectures)
- [Experimental Results](#-experimental-results)
- [Author & Contact](#-author--contact)
- [License](#-license)

---

## 📖 Abstract
With the rapid proliferation of Generative AI (GANs, Diffusion Models), the threat of **Deepfakes** and **Face Presentation Attacks** has escalated, posing severe risks to biometric security systems.

This repository hosts the implementation of my university research project at **Ho Chi Minh City University of Technology and Education (HCMUTE)**. The core contribution is a custom lightweight CNN architecture (**MyNet**) designed to efficiently distinguish between *bona fide* (real) faces and *spoof* (fake) faces. The project also provides a rigorous comparative analysis against heavyweight backbones like **MobileNetV2**, **ResNet50**, and **VGG16**.

## 🚀 Key Features
* **Custom "MyNet" Architecture:** A specialized CNN designed for spatial feature extraction with low computational cost.
* **Robust Preprocessing:** Automated face cropping, resizing, and data augmentation (Rotation, Zoom, Brightness, Flip) to handle diverse lighting conditions.
* **Experiment Tracking:** Built-in logging for Accuracy, Loss, Precision, Recall, and F1-Score during training.
* **Visualization:** Automatic generation of Confusion Matrices and ROC-AUC curves.
* **Modular Design:** Easy to swap backbones or add new datasets.

## 📂 Project Structure

```bash
CNNs-Deepfake_Detection/
├── data/
│   ├── raw/                 # Raw downloaded datasets
│   └── processed/           # Organized into Train/Val/Test
│       ├── train/           # 70% of data
│       ├── val/             # 15% of data
│       └── test/            # 15% of data
├── models/                  # Directory for saving model checkpoints (.h5)
├── notebooks/               # Jupyter Notebooks for EDA and rapid prototyping
├── results/                 # Output graphs, confusion matrices, and logs
├── src/                     # Source Code
│   ├── __init__.py
│   ├── config.py            # Hyperparameters configuration
│   ├── data_loader.py       # Data generators and preprocessing
│   ├── models.py            # Implementation of MyNet and Transfer Learning models
│   ├── train.py             # Main training loop
│   ├── evaluate.py          # Testing and metrics calculation
│   └── utils.py             # Helper functions (visualization, file handling)
├── requirements.txt         # Python dependencies list
├── LICENSE                  # MIT License
└── README.md                # Project documentation
