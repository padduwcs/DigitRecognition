import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import process
import predict
import numpy as np

k_values = [3,6,24]
extract_methods = {
    0: "FLAT",
    1: "CHUNK",
    2: "HISTOGRAM"
}

def fix(raw_image):
    image = raw_image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    st.image(image, caption="Received image")

    image_array = np.array(image)
    image_array = image_array.astype(np.float64)
    image_array = 255 - image_array

    return image_array

@st.cache_data
def load_data():
    x_train, y_train = process.load_mnist("data/", kind="train")
    train_flat, train_chunk, train_histogram = process.extract_features(x_train)
    
    combined_train_flat = process.combine(train_flat, y_train)
    combined_train_chunk = process.combine(train_chunk, y_train)
    combined_train_histogram = process.combine(train_histogram, y_train)
    return combined_train_flat, combined_train_chunk, combined_train_histogram

def solve(image_aray):
    combined_train_flat, combined_train_chunk, combined_train_histogram = load_data()
    results = predict.predict_with_methods(image_array, extract_methods, k_values, combined_train_flat, combined_train_chunk, combined_train_histogram)

    for method_name, answer in results:
        st.write(f"{method_name}'s prediction: {answer}")

st.title("Digit Recognition")

st.sidebar.title("Options") 
option = st.sidebar.radio("Choose an option:", ("Upload Image", "Draw"))

if option == "Upload Image":
    left, right = st.columns(2)

    with left:
        st.header('Upload a Digit Image')
        uploaded_file = st.file_uploader("Add image", type=["jpg", "jpeg", "png"])

    with right:
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            image_array = fix(raw_image)
            # st.write(image_array)

            if st.button("Submit"):
                solve(image_array)

if option == "Draw":
    left, right = st.columns(2)

    with left:
        st.header('Draw a Digit')

        canvas_result = st_canvas(
            fill_color          =   "rgba(255, 255, 255, 1)",  # Nền màu trắng
            stroke_width        =   25,                        # Độ dày nét vẽ
            stroke_color        =   "#000000",                 # Màu nét vẽ (đen)
            background_color    =   "#FFFFFF",                 # Nền canvas màu trắng
            width               =   280,                       # Chiều rộng của canvas
            height              =   280,                       # Chiều cao của canvas
            drawing_mode        =   "freedraw",                # Chế độ vẽ tự do
            key                 =   "canvas"                   # Khóa cho canvas
        )

    with right:
        if canvas_result.image_data is not None:
            raw_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            image_array = fix(raw_image)
            # st.write(image_array)

            if st.button("Submit"):
                solve(image_array)
