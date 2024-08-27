import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import pickle

# Path to the trained model
model_path = r'C:\Users\User\Desktop\Niqo-Robotics\linear_svc_pipeline.pkl'

def load_model(model_path):
    """Load the trained model from a file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)  # Load and return the model object

def preprocess_image(image):
    """Preprocess image for model prediction."""
    # Convert image to RGB and resize to 224x224 pixels
    img = image.convert('RGB').resize((224, 224))
    # Convert image to numpy array and normalize pixel values to [0, 1]
    img_array = np.array(img) / 255.0
    # Flatten the image array and reshape it to fit the model input
    img_flat = img_array.flatten().reshape(1, -1)
    return img_flat

def load_templates(dataset_dir, class_names):
    """Load template images for each class from the dataset directory."""
    templates = {}
    for class_name in class_names:
        # Construct the path for the template image
        template_path = os.path.join(dataset_dir, class_name, 'img0.jpg')  # Adjust file format if needed
        # Check if the template image exists and load it in grayscale
        if os.path.exists(template_path):
            templates[class_name] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: Template image does not exist at {template_path}")
    return templates

from imutils.object_detection import non_max_suppression
import logging

def segment_with_template_matching(image, templates):
    """Segment regions using template matching."""
    # Convert the uploaded image to grayscale
    image_gray = ImageOps.grayscale(image)
    img_np = np.array(image_gray)  # Convert grayscale image to numpy array

    results = []

    # Iterate over each template to perform template matching
    for class_name, template in templates.items():
        # Perform template matching
        result = cv2.matchTemplate(img_np, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5  # Set a threshold for detecting matches
        loc = np.where(result >= threshold)  # Find locations of matches above the threshold

        # Iterate over detected locations
        for pt in zip(*loc[::-1]):  # Reverse to (x, y)
            x, y = pt
            w, h = template.shape[::-1]  # Width and height of the template
            x1, y1 = x + w, y + h  # Calculate the bottom-right corner of the bounding box
            results.append({
                'bounding_box': (x, y, x1, y1),
                'class_name': class_name,
                'confidence': result[y, x]  # Confidence score of the match
            })

    return results, img_np

def predict_segments(image, segments, model, class_names):
    """Predict class for each segmented region."""
    predictions = []
    for segment in segments:
        x, y, x1, y1 = segment['bounding_box']
        # Crop the image to the bounding box and convert it to RGB
        cropped_image = image.crop((x, y, x1, y1)).convert('RGB')
        # Preprocess the cropped image
        img_flat = preprocess_image(cropped_image)
        # Predict the class of the cropped image
        prediction = model.predict(img_flat)[0]
        class_name = class_names[prediction]
        predictions.append({
            'bounding_box': segment['bounding_box'],
            'class_name': class_name,
            'confidence': segment['confidence']
        })
    return predictions
