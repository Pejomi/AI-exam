import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Traffic Sign Recognition")
model = YOLO("traffic_sign_model.pt")

if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None


def predict(input_image):
    prediction = model(input_image)
    prediction[0].save("prediction.jpg")
    output_img = Image.open("prediction.jpg").convert('RGB')
    os.remove("prediction.jpg")

    return output_img


# Specify the directory containing the pre-defined images
image_dir = '../img/traffic_images'

# List all files in the directory
files = os.listdir(image_dir)

# Filter for image files (assuming JPEG and PNG) and open them
images = [Image.open(os.path.join(image_dir, file)) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

css = '''
<style>
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

st.write("Select a sample image or upload your own to perform inference using the traffic sign detection model:")

cols = st.columns(len(images) + 1)  # Create columns for images + one for upload

for i, img in enumerate(images):
    with cols[i]:
        st.image(img, use_column_width=True, output_format='PNG', caption=f"Sample {i + 1}")
        if st.button(f"Select sample {i + 1}", key=f"select_{i}"):
            st.session_state.selected_image = img

# File uploader in the last column
with cols[-1]:
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key="uploader")


if st.session_state.selected_image or uploaded_file:
    display_image = st.session_state.selected_image if st.session_state.selected_image else Image.open(
        uploaded_file).convert('RGB')

    output_image = predict(display_image)
    st.image(output_image, caption='Predicted Image', use_column_width=True)


st.text("")
st.text("")
st.text("")
st.text("")
st.title("Info about the model:")
st.write("The model used for this task is the YOLOv8n model trained on 30,000 synthetic images of traffic-environments"
         " with appearing traffic signs. The images were created and generated using the Unity Game Engine and its Perception Engine package."
         " The model was trained for 30 epochs with a batch size of 64, and was trained on the NVIDIA 1070 Ti GPU.")
st.write(" A thing to note about the model's performance metrics, is that they are measured based on the synthetic images, and not real-world images."
         " This means that the model might not perform as well on real-world images as it does on the synthetic images.")

st.write("")
st.write("")
st.write("Below are the training metrics of the model, showing the loss and mAP (mean Average Precision) over the 30 epochs:")
st.image("../img/results.png", caption="Model Training Metrics", use_column_width=True)

st.write("")
st.write("")
st.write("And this is the produced confusion matrix, showing that the model has a high accuracy on the test set, with few misclassifications:")
st.image("../img/confusion_matrix_normalized.png", caption="Model Confusion Matrix (Normalized)", use_column_width=True)