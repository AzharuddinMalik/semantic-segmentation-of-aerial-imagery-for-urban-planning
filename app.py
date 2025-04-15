# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Custom Loss Functions and Metrics
def jacard_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = (intersection + 1e-15) / (sum_ - intersection + 1e-15)
    return jac

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fixed

def dice_loss_plus_1focal_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss()(y_true, y_pred)

model = load_model("models/satellite_standard_unet_100epochs_7May2021.hdf5",
                   custom_objects={
                       'jacard_coef': jacard_coef,
                       'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss
                   })

PATCH_SIZE = 256
N_CLASSES = 6
CLASS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (128, 0, 128), (255, 165, 0)]


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_segmentation(image_array):
    pred = model.predict(image_array)
    pred_mask = np.argmax(pred[0], axis=-1)
    confidence = np.max(pred[0], axis=-1)
    return pred_mask, confidence

def create_colored_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(N_CLASSES):
        color_mask[mask == i] = CLASS_COLORS[i]
    return color_mask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_array = preprocess_image(filepath)
        pred_mask, confidence = predict_segmentation(image_array)

        mask_colored = create_colored_mask(pred_mask)
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
        cv2.imwrite(mask_path, mask_colored)

        confidence_map = np.uint8(confidence * 255)
        confidence_path = os.path.join(app.config['UPLOAD_FOLDER'], 'confidence_' + filename)
        cv2.imwrite(confidence_path, confidence_map)

        return render_template('result.html',
                               input_image=url_for('static', filename='uploads/' + filename),
                               output_mask=url_for('static', filename='uploads/mask_' + filename),
                               confidence_map=url_for('static', filename='uploads/confidence_' + filename))

if __name__ == '__main__':
    app.run(debug=True)
