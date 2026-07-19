import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from .gradcam import make_gradcam_heatmap, overlay_gradcam
import cv2

# Define class names for HAM10000 dataset roughly
CLASS_NAMES = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)'
}

# Define risk levels matching classes (simplified for demonstration)
RISK_LEVELS = {
    0: 'Medium',  # Pre-cancerous
    1: 'High',    # Cancer
    2: 'Low',     # Benign
    3: 'Low',     # Benign
    4: 'High',    # Cancer (Melanoma)
    5: 'Low',     # Benign (Moles)
    6: 'Low'      # Benign
}

def load_dermascan_model(model_path='models/dermascan_model.h5'):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # For development, if no model trained yet, we will mock it
        print("Warning: Model not found. Returning a mock model/result.")
        return None

def predict_image(image_path, model=None, output_heatmap_path=None):
    """
    Loads an image, makes a prediction, and optionally generates a Grad-CAM heatmap.
    """
    # Use standard target size
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0

    if model is None:
        # Mock prediction for development
        pred_idx = np.random.randint(0, 7)
        confidence = float(np.random.uniform(0.6, 0.99))
        
        # Mock heatmap
        if output_heatmap_path:
            orig = cv2.imread(image_path)
            if orig is not None:
                heatmap = np.random.rand(224, 224)
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_color = cv2.resize(heatmap_color, (orig.shape[1], orig.shape[0]))
                superimposed = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
                cv2.imwrite(output_heatmap_path, superimposed)
            
        return {
            'class_index': pred_idx,
            'class_name': CLASS_NAMES[pred_idx],
            'confidence': confidence,
            'risk_level': RISK_LEVELS[pred_idx],
            'is_mock': True
        }
    
    # Real Inference
    preds = model.predict(img_array_expanded)
    pred_idx = np.argmax(preds[0])
    confidence = float(preds[0][pred_idx])

    # Try to generate Grad-CAM
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if output_heatmap_path and last_conv_layer_name:
        try:
            heatmap = make_gradcam_heatmap(img_array_expanded, model, last_conv_layer_name)
            superimposed = overlay_gradcam(image_path, heatmap)
            # overlay_gradcam returns RGB, imwrite needs BGR
            cv2.imwrite(output_heatmap_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")

    return {
        'class_index': int(pred_idx),
        'class_name': CLASS_NAMES[int(pred_idx)],
        'confidence': confidence,
        'risk_level': RISK_LEVELS[int(pred_idx)],
        'is_mock': False
    }
