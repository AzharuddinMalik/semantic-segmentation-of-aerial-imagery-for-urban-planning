# Semantic Segmentation of Aerial Imagery 🌍

This project focuses on pixel-wise classification of aerial/satellite imagery using deep learning models like U-Net and ResNet-based UNet. The goal is to extract meaningful land features such as buildings, roads, water, and vegetation for urban planning, agriculture, and environmental analysis.

## 🚀 Project Overview

- Segment satellite images into 6 categories: Buildings, Land, Roads, Vegetation, Water, and Unlabeled areas.
- Trained using both custom and pretrained deep learning models (U-Net, ResNet34-UNet).
- Web demo built with Flask for interactive predictions.

## 📊 Dataset

Dataset from [Humans in the Loop (Kaggle)](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) containing high-resolution satellite imagery of Dubai, annotated with semantic classes.

## 🛠 Tools & Technologies

- Python, Keras, TensorFlow, OpenCV
- Flask (for deployment)
- Patchify, Albumentations (for data preparation & augmentation)
- Matplotlib, NumPy, PIL, Scikit-learn
- Jupyter Notebook, VS Code, GitHub

## ✨ Features

- 🔍 Preprocessing: Patchify aerial tiles to 256×256 segments.
- 🧠 Training: Uses Dice + Focal loss with class weighting.
- 📈 Evaluation: Mean IoU, Dice Coefficient, Accuracy.
- 🖼 Visualization: Overlay masks & confidence maps.
- 🌐 Web App: Upload aerial images and get segmented masks instantly.

## 🌟 Advantages

- High precision in detecting complex land patterns.
- Easily extendable to new datasets or custom label schemes.
- Lightweight web interface for real-time predictions.
- Integrates both standard and pretrained segmentation pipelines.

## 📦 Project Structure

  - app.py # Flask Web Application.
  - training_aerial_imagery.py # Model training script.
  - simple_multi_unet_model.py # U-Net model definition.
  -  static/uploads/ # Stores uploaded & predicted images.
  -   models/ # Trained model .hdf5 file.
  -   templates/ # HTML templates.
  -   README.md

## 🚀 Running the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the web app:
    ```bash
    python app.py
3.Open your browser at http://127.0.0.1:5000/ and upload an image to see predictions.

