import os
import io
import time
import numpy as np
import pandas as pd
import cv2
import joblib
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ----------------- Configuration -----------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
STATIC_FOLDER = os.path.join(APP_ROOT, 'static')
TEMP_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, 'session_images')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB limit

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Model file paths
CNN_MODEL_PATH = os.path.join(APP_ROOT, 'models', '96_best_brain_tumor_model.keras')
RF_MODEL_PATH = os.path.join(APP_ROOT, 'models', 'brain_tumor_rf_model.pkl')
SCALER_PATH = os.path.join(APP_ROOT, 'models', 'scaler.pkl')
UNET_MODEL_PATH = os.path.join(APP_ROOT, 'models', 'unet_brain_tumor.h5')

# App constants
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CNN_IMAGE_SIZE = (224, 224)  # (width, height)
UNET_IMAGE_SIZE = (128, 128)
NUM_CLASSES_UNET = 4
CNN_WEIGHT = 0.7
ML_WEIGHT = 0.3
LAST_CONV_LAYER_NAME = 'conv5_block3_out'
PREPROCESS_FN = resnet_preprocess
HEADACHE_MAPPING = {'No': 0, 'Low': 1, 'Mild': 2, 'Severe': 3}

# --- Custom Loss for U-Net ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_temp_images(age_seconds=3600):
    now = time.time()
    for fname in os.listdir(TEMP_IMAGE_FOLDER):
        fpath = os.path.join(TEMP_IMAGE_FOLDER, fname)
        try:
            if os.path.isfile(fpath):
                if now - os.path.getmtime(fpath) > age_seconds:
                    os.remove(fpath)
        except Exception:
            pass

# --- Load models ---
cnn_model = None
rf_model = None
scaler = None
unet_model = None

def load_all_models():
    global cnn_model, rf_model, scaler, unet_model
    try:
        cnn_model = load_model(CNN_MODEL_PATH)
        print(f"[INFO] Loaded CNN model from: {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load CNN model: {e}")
        cnn_model = None

    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"[INFO] Loaded RF model from: {RF_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load RF model: {e}")
        rf_model = None

    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"[INFO] Loaded scaler from: {SCALER_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load scaler: {e}")
        scaler = None

    try:
        with tf.keras.utils.custom_object_scope({'combined_loss': combined_loss, 'dice_loss': dice_loss}):
            unet_model = load_model(UNET_MODEL_PATH, compile=False)
        print(f"[INFO] Loaded U-Net model from: {UNET_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load U-Net model: {e}")
        unet_model = None

load_all_models()

# --- Grad-CAM ---
def get_grad_cam_heatmap(model, img, last_conv_layer_name, original_size):
    if model is None:
        return np.zeros(original_size[::-1], dtype=np.float32)

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        print(f"[Grad-CAM] Layer not found: {e}")
        for layer in reversed(model.layers):
            if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                print(f"[Grad-CAM] Fallback to layer: {last_conv_layer_name}")
                break
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except Exception:
            return np.zeros(original_size[::-1], dtype=np.float32)

    img_batch = np.expand_dims(img.astype(np.float32), axis=0)
    img_pre = PREPROCESS_FN(img_batch)

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_pre])
        top_class = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_class]

    grads = tape.gradient(top_class_channel, conv_outputs)
    if grads is None:
        print("[Grad-CAM] grads is None")
        return np.zeros(original_size[::-1], dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap /= max_val
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], original_size[::-1])
    return np.squeeze(heatmap.numpy())

# --- U-Net Segmentation ---
def get_segmentation_mask(image_path, target_size=UNET_IMAGE_SIZE):
    if unet_model is None:
        print("[WARN] U-Net model is not loaded. Cannot perform segmentation.")
        return None

    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        img_resized = cv2.resize(img, target_size)
        img_normalized = np.expand_dims(img_resized, axis=0) / 255.0

        prediction = unet_model.predict([img_normalized], verbose=0)
        predicted_mask = np.argmax(prediction[0], axis=-1)
        return predicted_mask.astype(np.uint8)

    except Exception as e:
        print(f"[U-Net] Segmentation failed: {e}")
        return None

# --- Prediction logic for different modes ---
def predict_symptoms_only(patient_data):
    """
    patient_data here is expected to contain only model features (NOT name/mobile).
    """
    if rf_model is None or scaler is None:
        return "Models not loaded", None, None

    symptom_df = pd.DataFrame([patient_data])
    mapping_gender = {'Female': 0, 'Male': 1}
    mapping_yesno = {'No': 0, 'Yes': 1}

    try:
        symptom_df['Gender'] = symptom_df['Gender'].map(mapping_gender)
        symptom_df['FamilyHistory'] = symptom_df['FamilyHistory'].map(mapping_yesno)
        symptom_df['HeadInjury'] = symptom_df['HeadInjury'].map(mapping_yesno)
        symptom_df['RadiationExposure'] = symptom_df['RadiationExposure'].map(mapping_yesno)
        symptom_df['Headache'] = symptom_df['Headache'].map(HEADACHE_MAPPING)
        symptom_df['Nausea'] = symptom_df['Nausea'].map(mapping_yesno)
        symptom_df['Mood'] = symptom_df['Mood'].map(mapping_yesno)
        symptom_df['Cognitive'] = symptom_df['Cognitive'].map(mapping_yesno)
        symptom_df['Speech'] = symptom_df['Speech'].map(mapping_yesno)
        symptom_df['Seizures'] = symptom_df['Seizures'].map(mapping_yesno)
        symptom_df['Vision'] = symptom_df['Vision'].map(mapping_yesno)
        symptom_df['Hearing'] = symptom_df['Hearing'].map(mapping_yesno)
        symptom_df['Balance'] = symptom_df['Balance'].map(mapping_yesno)
        symptom_df['Sensory'] = symptom_df['Sensory'].map(mapping_yesno)
    except Exception as e:
        return f"Bad symptom input: {e}", None, None

    ml_features = ['Headache', 'Nausea', 'Mood', 'Cognitive', 'Speech', 'Seizures', 'Vision', 'Hearing', 'Balance',
                   'Sensory', 'Age', 'Gender', 'FamilyHistory', 'HeadInjury', 'RadiationExposure']
    symptom_df = symptom_df.reindex(columns=ml_features)

    try:
        symptom_data_scaled = scaler.transform(symptom_df)
        proba = rf_model.predict_proba(symptom_data_scaled)[0]
    except Exception as e:
        return f"Model/scaler error: {e}", None, None

    if len(proba) == 2:
        ml_prediction_prob = proba[1]
    else:
        ml_prediction_prob = 1.0 - proba[CLASS_NAMES.index('notumor')]

    final_class_name = 'No'
    if ml_prediction_prob >= 0.5:
        final_class_name = 'Yes'

    return final_class_name, {CLASS_NAMES[i]: f'{p:.4f}' for i, p in enumerate(proba)}, ml_prediction_prob

def predict_mri_only(image_path):
    if cnn_model is None:
        return "Models not loaded", None, None, None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "Image not found", None, None, None

    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_size = (img_rgb_original.shape[1], img_rgb_original.shape[0])
    img_resized_for_cnn = cv2.resize(img_rgb_original, CNN_IMAGE_SIZE)

    try:
        img_pre = PREPROCESS_FN(np.expand_dims(img_resized_for_cnn.astype(np.float32), axis=0))
        cnn_probs = cnn_model.predict([img_pre], verbose=0)[0]
    except Exception as e:
        return f"CNN prediction failed: {e}", None, None, None

    heatmap = get_grad_cam_heatmap(cnn_model, img_resized_for_cnn, LAST_CONV_LAYER_NAME, original_size)
    segmentation_mask = get_segmentation_mask(image_path)

    final_idx = int(np.argmax(cnn_probs))
    final_name = CLASS_NAMES[final_idx]

    return final_name, {CLASS_NAMES[i]: f'{p:.4f}' for i, p in enumerate(cnn_probs)}, heatmap, segmentation_mask

def predict_combined(patient_data, image_path):
    if cnn_model is None or rf_model is None or scaler is None:
        return "Models not loaded", None, None, None

    # --- Symptoms ---
    symptom_df = pd.DataFrame([patient_data])
    mapping_gender = {'Female': 0, 'Male': 1}
    mapping_yesno = {'No': 0, 'Yes': 1}

    try:
        symptom_df['Gender'] = symptom_df['Gender'].map(mapping_gender)
        symptom_df['FamilyHistory'] = symptom_df['FamilyHistory'].map(mapping_yesno)
        symptom_df['HeadInjury'] = symptom_df['HeadInjury'].map(mapping_yesno)
        symptom_df['RadiationExposure'] = symptom_df['RadiationExposure'].map(mapping_yesno)
        symptom_df['Headache'] = symptom_df['Headache'].map(HEADACHE_MAPPING)
        symptom_df['Nausea'] = symptom_df['Nausea'].map(mapping_yesno)
        symptom_df['Mood'] = symptom_df['Mood'].map(mapping_yesno)
        symptom_df['Cognitive'] = symptom_df['Cognitive'].map(mapping_yesno)
        symptom_df['Speech'] = symptom_df['Speech'].map(mapping_yesno)
        symptom_df['Seizures'] = symptom_df['Seizures'].map(mapping_yesno)
        symptom_df['Vision'] = symptom_df['Vision'].map(mapping_yesno)
        symptom_df['Hearing'] = symptom_df['Hearing'].map(mapping_yesno)
        symptom_df['Balance'] = symptom_df['Balance'].map(mapping_yesno)
        symptom_df['Sensory'] = symptom_df['Sensory'].map(mapping_yesno)
    except Exception as e:
        return f"Bad symptom input: {e}", None, None, None

    ml_features = ['Headache', 'Nausea', 'Mood', 'Cognitive', 'Speech', 'Seizures', 'Vision', 'Hearing', 'Balance',
                   'Sensory', 'Age', 'Gender', 'FamilyHistory', 'HeadInjury', 'RadiationExposure']
    symptom_df = symptom_df.reindex(columns=ml_features)

    try:
        symptom_data_scaled = scaler.transform(symptom_df)
        proba = rf_model.predict_proba(symptom_data_scaled)[0]
    except Exception as e:
        return f"Model/scaler error: {e}", None, None, None

    ml_probs = np.zeros(len(CLASS_NAMES), dtype=float)
    if len(proba) == 2:
        ml_prediction_prob = proba[1]
    else:
        ml_prediction_prob = 1.0 - proba[CLASS_NAMES.index('notumor')]

    if ml_prediction_prob >= 0.5:
        tumor_idxs = [i for i, n in enumerate(CLASS_NAMES) if n != 'notumor']
        for idx in tumor_idxs:
            ml_probs[idx] = ML_WEIGHT / len(tumor_idxs)
    else:
        ml_probs[CLASS_NAMES.index('notumor')] = ML_WEIGHT

    # --- MRI Image Analysis ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "Image not found", None, None, None

    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_size = (img_rgb_original.shape[1], img_rgb_original.shape[0])
    img_resized_for_cnn = cv2.resize(img_rgb_original, CNN_IMAGE_SIZE)

    try:
        img_pre = PREPROCESS_FN(np.expand_dims(img_resized_for_cnn.astype(np.float32), axis=0))
        cnn_probs = cnn_model.predict([img_pre], verbose=0)[0]
    except Exception as e:
        return f"CNN prediction failed: {e}", None, None, None

    heatmap = get_grad_cam_heatmap(cnn_model, img_resized_for_cnn, LAST_CONV_LAYER_NAME, original_size)
    segmentation_mask = get_segmentation_mask(image_path)

    # --- Combine ---
    combined_probs = (cnn_probs * CNN_WEIGHT) + ml_probs
    if combined_probs.sum() > 0:
        combined_probs = combined_probs / combined_probs.sum()

    final_idx = int(np.argmax(combined_probs))
    final_name = CLASS_NAMES[final_idx]

    return final_name, {CLASS_NAMES[i]: f'{p:.4f}' for i, p in enumerate(combined_probs)}, heatmap, segmentation_mask

# ----------------- Flask routes -----------------
@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/symptoms')
def symptoms_page():
    return render_template('symptoms.html')

@app.route('/mri_upload')
def mri_upload_page():
    return render_template('mri_upload.html')

@app.route('/combined')
def combined_page():
    return render_template('combined.html')

@app.route('/predict_symptoms', methods=['POST'])
def handle_symptoms_only_request():
    """
    Accepts patient details + symptom fields from the form.
    Stores:
      - session['patient_details']  (Name, Age, Gender, Mobile)
      - session['patient_data']     (model features only)
    """
    try:
        # --- Patient details (for display only) ---
        patient_details = {
            'Name': request.form.get('name', '').strip(),
            'Age': int(request.form.get('age', 0)),
            'Gender': request.form.get('gender', 'Male'),
            'Mobile': request.form.get('mobile', '').strip()
        }

        # Basic server-side validation
        if not patient_details['Name'] or patient_details['Age'] <= 0 or patient_details['Age'] > 120:
            return jsonify({'error': 'Invalid name or age provided.'}), 400
        if patient_details['Gender'] not in ['Male', 'Female']:
            return jsonify({'error': 'Invalid gender value.'}), 400
        if not (patient_details['Mobile'].isdigit() and len(patient_details['Mobile']) == 10):
            return jsonify({'error': 'Mobile number must be 10 digits.'}), 400

        # --- Features for model (do NOT include name/mobile) ---
        symptom_data = {
            'Age': patient_details['Age'],
            'Gender': patient_details['Gender'],
            'FamilyHistory': request.form.get('familyHistory', 'No'),
            'HeadInjury': request.form.get('headInjury', 'No'),
            'RadiationExposure': request.form.get('radiationExposure', 'No'),
            'Headache': request.form.get('headache', 'No'),
            'Nausea': request.form.get('nausea', 'No'),
            'Mood': request.form.get('mood', 'No'),
            'Cognitive': request.form.get('cognitive', 'No'),
            'Speech': request.form.get('speech', 'No'),
            'Seizures': request.form.get('seizures', 'No'),
            'Vision': request.form.get('vision', 'No'),
            'Hearing': request.form.get('hearing', 'No'),
            'Balance': request.form.get('balance', 'No'),
            'Sensory': request.form.get('sensory', 'No')
        }
    except Exception as e:
        return jsonify({'error': f'Invalid form data: {e}'}), 400

    final_class, probabilities, ml_prediction_prob = predict_symptoms_only(symptom_data)

    if final_class in ["Models not loaded", "Bad symptom input", "Model/scaler error"]:
        return jsonify({'error': final_class}), 500

    # Save to session for report
    session['patient_details'] = patient_details
    session['patient_data'] = symptom_data
    session['final_class'] = final_class
    session['ml_prediction_prob'] = float(ml_prediction_prob)
    session['analysis_mode'] = 'symptoms'
    return jsonify({'redirect_url': url_for('show_report')}), 200

@app.route('/predict_mri', methods=['POST'])
def handle_mri_only_request():
    try:
        # --- Patient details (for display only) ---
        patient_details = {
            'Name': request.form.get('name', '').strip(),
            'Age': int(request.form.get('age', 0)),
            'Gender': request.form.get('gender', 'Male'),
            'Mobile': request.form.get('mobile', '').strip()
        }

        # Basic server-side validation
        if not patient_details['Name'] or patient_details['Age'] <= 0 or patient_details['Age'] > 120:
            return jsonify({'error': 'Invalid name or age provided.'}), 400
        if patient_details['Gender'] not in ['Male', 'Female']:
            return jsonify({'error': 'Invalid gender value.'}), 400
        if not (patient_details['Mobile'].isdigit() and len(patient_details['Mobile']) == 10):
            return jsonify({'error': 'Mobile number must be 10 digits.'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid form data: {e}'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save upload: {e}'}), 500

    final_class, probabilities, heatmap, segmentation_mask = predict_mri_only(filepath)

    if final_class in ["Models not loaded", "Image not found", "CNN prediction failed"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({'error': final_class}), 500

    try:
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_size = (img_rgb.shape[1], img_rgb.shape[0])

        if heatmap is not None:
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(cv2.resize(heatmap_uint8, original_size), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            superimposed_heatmap = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
        else:
            superimposed_heatmap = img_rgb

        if segmentation_mask is not None:
            segmentation_mask_resized = cv2.resize(segmentation_mask, original_size, interpolation=cv2.INTER_NEAREST)
            cmap = plt.get_cmap('jet', NUM_CLASSES_UNET)
            seg_colored = (cmap(segmentation_mask_resized)[..., :3] * 255).astype(np.uint8)
            superimposed_segmentation = cv2.addWeighted(img_rgb, 0.6, seg_colored, 0.4, 0)
        else:
            superimposed_segmentation = img_rgb

    except Exception as e:
        print(f"[Image prepare] {e}")
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        superimposed_heatmap = img_rgb
        superimposed_segmentation = img_rgb

    try:
        os.remove(filepath)
    except Exception:
        pass

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    orig_fname = f"orig_{timestamp}.png"
    hm_fname = f"heatmap_{timestamp}.png"
    seg_fname = f"segmentation_{timestamp}.png"

    orig_path = os.path.join(TEMP_IMAGE_FOLDER, orig_fname)
    hm_path = os.path.join(TEMP_IMAGE_FOLDER, hm_fname)
    seg_path = os.path.join(TEMP_IMAGE_FOLDER, seg_fname)

    try:
        Image.fromarray(img_rgb).save(orig_path)
        Image.fromarray(superimposed_heatmap).save(hm_path)
        Image.fromarray(superimposed_segmentation).save(seg_path)
    except Exception as e:
        print(f"[Save temp images] {e}")

    session['patient_details'] = patient_details
    session['original_image_file'] = orig_fname
    session['heatmap_file'] = hm_fname
    session['segmentation_file'] = seg_fname
    session['final_class'] = final_class
    session['probabilities'] = probabilities
    session['analysis_mode'] = 'mri'
    cleanup_old_temp_images(age_seconds=3600)
    return jsonify({'redirect_url': url_for('show_report')}), 200

@app.route('/predict_combined', methods=['POST'])
def handle_combined_request():
    try:
        # --- Patient details (for display only) ---
        patient_details = {
            'Name': request.form.get('name', '').strip(),
            'Age': int(request.form.get('age', 0)),
            'Gender': request.form.get('gender', 'Male'),
            'Mobile': request.form.get('mobile', '').strip()
        }
        # Basic server-side validation for patient details
        if not patient_details['Name'] or patient_details['Age'] <= 0 or patient_details['Age'] > 120:
            return jsonify({'error': 'Invalid name or age provided.'}), 400
        if patient_details['Gender'] not in ['Male', 'Female']:
            return jsonify({'error': 'Invalid gender value.'}), 400
        if not (patient_details['Mobile'].isdigit() and len(patient_details['Mobile']) == 10):
            return jsonify({'error': 'Mobile number must be 10 digits.'}), 400
    except Exception as e:
        return jsonify({'error': f'Invalid patient data: {e}'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save upload: {e}'}), 500

    try:
        symptom_data = {
            'Age': int(request.form.get('age', 0)),
            'Gender': request.form.get('gender', 'Male'),
            'FamilyHistory': request.form.get('familyHistory', 'No'),
            'HeadInjury': request.form.get('headInjury', 'No'),
            'RadiationExposure': request.form.get('radiationExposure', 'No'),
            'Headache': request.form.get('headache', 'No'),
            'Nausea': request.form.get('nausea', 'No'),
            'Mood': request.form.get('mood', 'No'),
            'Cognitive': request.form.get('cognitive', 'No'),
            'Speech': request.form.get('speech', 'No'),
            'Seizures': request.form.get('seizures', 'No'),
            'Vision': request.form.get('vision', 'No'),
            'Hearing': request.form.get('hearing', 'No'),
            'Balance': request.form.get('balance', 'No'),
            'Sensory': request.form.get('sensory', 'No')
        }
    except Exception as e:
        os.remove(filepath)
        return jsonify({'error': f'Invalid form data: {e}'}), 400

    final_class, probabilities, heatmap, segmentation_mask = predict_combined(symptom_data, filepath)

    if final_class in ["Models not loaded", "Image not found", "Model/scaler error", "CNN prediction failed",
                       "Bad symptom input"]:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({'error': final_class}), 500

    try:
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_size = (img_rgb.shape[1], img_rgb.shape[0])

        if heatmap is not None:
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(cv2.resize(heatmap_uint8, original_size), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            superimposed_heatmap = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
        else:
            superimposed_heatmap = img_rgb

        if segmentation_mask is not None:
            segmentation_mask_resized = cv2.resize(segmentation_mask, original_size, interpolation=cv2.INTER_NEAREST)
            cmap = plt.get_cmap('jet', NUM_CLASSES_UNET)
            seg_colored = (cmap(segmentation_mask_resized)[..., :3] * 255).astype(np.uint8)
            superimposed_segmentation = cv2.addWeighted(img_rgb, 0.6, seg_colored, 0.4, 0)
        else:
            superimposed_segmentation = img_rgb

    except Exception as e:
        print(f"[Image prepare] {e}")
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        superimposed_heatmap = img_rgb
        superimposed_segmentation = img_rgb

    try:
        os.remove(filepath)
    except Exception:
        pass

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    orig_fname = f"orig_{timestamp}.png"
    hm_fname = f"heatmap_{timestamp}.png"
    seg_fname = f"segmentation_{timestamp}.png"

    orig_path = os.path.join(TEMP_IMAGE_FOLDER, orig_fname)
    hm_path = os.path.join(TEMP_IMAGE_FOLDER, hm_fname)
    seg_path = os.path.join(TEMP_IMAGE_FOLDER, seg_fname)

    try:
        Image.fromarray(img_rgb).save(orig_path)
        Image.fromarray(superimposed_heatmap).save(hm_path)
        Image.fromarray(superimposed_segmentation).save(seg_path)
    except Exception as e:
        print(f"[Save temp images] {e}")

    session['patient_details'] = patient_details
    session['patient_data'] = symptom_data
    session['original_image_file'] = orig_fname
    session['heatmap_file'] = hm_fname
    session['segmentation_file'] = seg_fname
    session['final_class'] = final_class
    session['probabilities'] = probabilities
    session['analysis_mode'] = 'combined'
    cleanup_old_temp_images(age_seconds=3600)
    return jsonify({'redirect_url': url_for('show_report')}), 200

@app.route('/report')
def show_report():
    if 'final_class' not in session:
        return redirect(url_for('landing_page'))

    analysis_mode = session.get('analysis_mode', 'combined')

    is_error = False
    if session.get('final_class') in ["Models not loaded", "Image not found", "Model/scaler error",
                                      "CNN prediction failed", "Bad symptom input"]:
        is_error = True

    report_data = {
        'analysis_mode': analysis_mode,
        'patient_details': session.get('patient_details') if analysis_mode in ['symptoms', 'mri', 'combined'] else None,
        'patient_data': session.get('patient_data') if analysis_mode in ['symptoms', 'combined'] else None,
        'original_image_url': url_for('static',
                                      filename=f'session_images/{session.get("original_image_file")}') if analysis_mode in [
            'mri', 'combined'] else None,
        'heatmap_url': url_for('static', filename=f'session_images/{session.get("heatmap_file")}') if analysis_mode in [
            'mri', 'combined'] else None,
        'segmentation_url': url_for('static',
                                    filename=f'session_images/{session.get("segmentation_file")}') if analysis_mode in [
            'mri', 'combined'] else None,
        'final_class': session.get('final_class'),
        'ml_prediction_prob': session.get('ml_prediction_prob'),
        'probabilities': session.get('probabilities'),
        'now': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if is_error:
        return render_template('error_report.html', data=report_data)

    if analysis_mode == 'symptoms':
        return render_template('symptoms_report.html', data=report_data)
    elif analysis_mode == 'mri':
        return render_template('mri_report.html', data=report_data)
    else:
        return render_template('combined_report.html', data=report_data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing_page'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)