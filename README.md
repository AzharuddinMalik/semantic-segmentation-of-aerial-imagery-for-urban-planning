# Semantic Segmentation of Aerial Imagery 🌍

This project focuses on pixel-wise classification of aerial/satellite imagery using deep learning models like U-Net and ResNet-based UNet. The goal is to extract meaningful land features such as buildings, roads, water, and vegetation for urban planning, agriculture, and environmental analysis.
<p align="center">
  <img src="IMAGES/Home Page.png" width="600"/>
</p>
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



📬 Contact
For questions or collaboration, feel free to connect via GitHub Issues or Pull Requests!


---

### 📜 **GitHub Project Description**
> Semantic segmentation of satellite imagery using deep learning (U-Net, ResNet34). Includes model training, evaluation, and a web app for real-time predictions.
<p align="center">
  <img src="IMAGES/About Project.png" width="600"/>
</p>
<p align="center">
  <img src="IMAGES/Gallary.png" width="600"/>
</p>
<p align="center">
  <img src="IMAGES/How It Works.png" width="600"/>
</p>
---

### ✅ **Feature & Advantages Summary**

**Features**
- Patch-wise training using `patchify`
- Dice + Focal loss combination
- Multi-class label segmentation (6 classes)
- Flask web app for deployment
- Custom and pretrained U-Net models

**Advantages**
- Accurate pixel-level classification
- Adaptable to different cities or satellite datasets
- Useful in urban planning, agriculture, and disaster response
- Easy-to-use web UI for demonstration

---

Let me know if you want me to generate the `requirements.txt` and `.gitignore` files for your repo as well!
