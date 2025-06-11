import numpy as np
from PIL import Image
import onnxruntime as ort
from django.conf import settings
import os

def preprocess_image(image_path):
    """Preprocess image for ONNX model."""
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    # Convert to numpy array and ensure float32 from the start
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Apply normalization with mean and std (ensure float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    # Transpose to CHW format (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Explicitly verify and enforce float32 type
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    
    return img_array

def run_prediction(image_path):
    """Run prediction using ONNX model."""
    model_path = os.path.join(settings.BASE_DIR, 'cattle_disease_model.onnx')
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Preprocess the image (already ensures float32 output)
    img_array = preprocess_image(image_path)

    # Debug: Verify data type and shape before inference
    print(f"Input tensor dtype: {img_array.dtype}, Shape: {img_array.shape}")

    # Ensure float32 type before passing to ONNX Runtime
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: img_array})
    probabilities = outputs[0][0]  # Assuming softmax output

    # Map probabilities to class names
    class_names = ["infected_foot", "infected_mouth"]
    result = {
        'infected_foot': probabilities[0] > 0.5,
        'infected_mouth': probabilities[1] > 0.5,
    }
    result['is_infected'] = result['infected_foot'] or result['infected_mouth']
    result['is_healthy'] = not result['is_infected']
    return result

def generate_explainability_image(image_path):
    """Placeholder for generating explainability image (e.g., Grad-CAM)."""
    # Implement Grad-CAM or similar technique using ONNX model
    # For simplicity, assume an explainability image is generated and saved
    explainability_path = image_path.replace('uploads', 'explainability')
    # Mock: Copy original image as explainability (replace with actual Grad-CAM logic)
    Image.open(image_path).save(explainability_path)
    return explainability_path