import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import os
import pickle
import tempfile
from segment import load_model, load_templates, segment_with_template_matching, predict_segments

# Load the SVC model
model_path = r'\Niqo-Robotics\linear_svc_pipeline.pkl'
svc_model = load_model(model_path)

# Class names for font classification
class_names = [
    "Arimo-Regular",
    "Dancing+Script-Regular",
    "FredokaOne-Regular",
    "NotoSans-Regular",
    "Open+Sans-Regular",
    "Oswald-Regular",
    "PTSerif-Regular",
    "PatuaOne-Regular",
    "Roboto-Regular",
    "Ubuntu-Regular"
]

def draw_bounding_boxes(image, results):
    """Draw bounding boxes on the image."""
    for result in results:
        x, y, x1, y1 = result['bounding_box']
        cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
    return image

def main():
    st.title("Font Identification")
    
    # Instructions for the user
    st.write("Upload an image, and the app will identify the fonts used in different regions.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file_path = temp_file.name
            # Save the uploaded image to the temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.read())
        
        # Load templates
        dataset_dir = r'\Niqo-Robotics\Dataset\images_fonts\images'
        templates = load_templates(dataset_dir, class_names)
        
        # Open and display the uploaded image
        image = Image.open(temp_file_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform template matching
        segments, img_np = segment_with_template_matching(image, templates)
        
        if not segments:
            st.warning("No text regions detected.")
        else:
            # Predict classes for segmented regions
            predictions = predict_segments(image, segments, svc_model, class_names)
            
            # Convert the grayscale image back to RGB for displaying in Streamlit
            image_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            # Draw bounding boxes on the image
            image_with_boxes = draw_bounding_boxes(image_rgb.copy(), predictions)
            
            # Display the image with bounding boxes
            st.image(image_with_boxes, caption='Detected Text Regions', use_column_width=True)
            
            # Display details for each detected region
            for i, prediction in enumerate(predictions):
                st.write(f"**Region {i+1}:**")
                st.write(f"**Bounding Box:** {prediction['bounding_box']}")
                st.write(f"**Predicted Font:** {prediction['class_name']}")
                st.write(f"**Confidence:** {prediction['confidence']:.2f}")

        # Clean up the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
