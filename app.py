# app.py
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash, abort, logging
import os

from tensorflow.python.keras.backend import clear_session
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from flask import jsonify  # Add this import at the top
from flask_wtf.csrf import CSRFProtect



app = Flask(__name__)


# Add after app initialization
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure this path matches Render's persistent storage
# app.config['UPLOAD_FOLDER'] = '/opt/render/project/src/static/uploads'

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

def load_model_safely():
    clear_session()
    return load_model(
        "models/satellite_standard_unet_100epochs_7May2021.hdf5",
        custom_objects={
            'jacard_coef': jacard_coef,
            'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss
        },
        compile=False  # Reduce memory footprint
    )

model = load_model_safely()

PATCH_SIZE = 256
N_CLASSES = 6
CLASS_COLORS = [
    (226, 169, 41),   # Water - #E2A929
    (254, 221, 58),   # Vegetation - #FEDD3A
    (110, 193, 228),  # Road - #6EC1E4
    (60, 16, 152),    # Building - #3C1098
    (132, 41, 246),   # Land - #8429F6
    (155, 155, 155)   # Unlabeled - #9B9B9B
]


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
    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)


PAGE_TITLE = "Welcome to My Website"


@app.route('/')
def render_home_page():
    return render_template('index.html', title=PAGE_TITLE)

# Add these routes to your existing app.py file

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')


@app.route('/result')
def result():
    input_image = request.args.get('input')
    mask_image = request.args.get('mask')
    confidence_map = request.args.get('confidence')

    # Ensure default values for missing data
    segmentation_data = {
        'area_distribution': [15, 25, 20, 30, 5, 5],  # Replace with actual data
        'class_coverage': [15, 25, 20, 30, 5, 5],
        'stats': {
            'total_area': '1.2 km²',
            'building_density': '30%',
            'green_space': '25%',
            'water_coverage': '15%',
            'road_density': '20%'
        }
    }

    return render_template('result.html',
                         input=input_image,
                         mask=mask_image,
                         confidence=confidence_map,
                         segmentation_data=segmentation_data)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if ('.' not in file.filename or
                file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions):
            return jsonify({'error': 'Invalid file type'}), 400

        # Create upload directory if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Generate unique filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_array = preprocess_image(filepath)
        pred_mask, confidence = predict_segmentation(image_array)

        mask_colored = create_colored_mask(pred_mask)
        mask_filename = 'mask_' + filename
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, mask_colored)

        confidence_filename = 'confidence_' + filename
        confidence_path = os.path.join(app.config['UPLOAD_FOLDER'], confidence_filename)
        cv2.imwrite(confidence_path, np.uint8(confidence * 255))

        return jsonify({
            'input_image': url_for('static', filename=f'uploads/{filename}'),
            'output_mask': url_for('static', filename=f'uploads/mask_{filename}'),
            'confidence_map': url_for('static', filename=f'uploads/confidence_{filename}')
        })


    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Add this new route for displaying results
@app.route('/result')
def show_result():
    input_image = request.args.get('input')
    output_mask = request.args.get('mask')
    confidence_map = request.args.get('confidence')

    if not all([input_image, output_mask, confidence_map]):
        abort(400, description="Missing required parameters")

    return render_template('result.html',
                           input_image=input_image,
                           output_mask=output_mask,
                           confidence_map=confidence_map)


if __name__ == '__main__':
    app.run()
