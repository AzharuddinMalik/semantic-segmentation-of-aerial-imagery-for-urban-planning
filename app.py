# app.py
import base64
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from flask import Flask, render_template, request, url_for, abort, logging
from flask import jsonify  # Add this import at the top
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

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
    K.clear_session()
    model = load_model(
        "models/satellite_standard_unet_100epochs_7May2021.hdf5",
        custom_objects={
            'jacard_coef': jacard_coef,
            'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss
        },
        compile=False
    )
    # Warm-up model
    model.predict(np.zeros((1, 256, 256, 3)))
    return model

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

        # Verify file content
        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid file type'}), 400

        # In-memory processing
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Process image
        img_resized = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
        image_array = np.expand_dims(img_resized / 255.0, axis=0)

        # Prediction
        pred_mask, confidence = predict_segmentation(image_array)
        mask_colored = create_colored_mask(pred_mask)

        # Convert to base64
        _, buffer_orig = cv2.imencode('.png', img)
        _, buffer_mask = cv2.imencode('.png', mask_colored)
        _, buffer_conf = cv2.imencode('.png', np.uint8(confidence * 255))

        return jsonify({
            'input_image': f'data:image/png;base64,{base64.b64encode(buffer_orig).decode("utf-8")}',
            'output_mask': f'data:image/png;base64,{base64.b64encode(buffer_mask).decode("utf-8")}',
            'confidence_map': f'data:image/png;base64,{base64.b64encode(buffer_conf).decode("utf-8")}'
        })

    except Exception as e:
        logging.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Image processing failed'}), 500


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
