import streamlit as st
from PIL import Image
import tempfile
from app import predict_top3  # importing your function

# Page config
st.set_page_config(
    page_title="🐶 Dog Breed Classifier",
    page_icon="🐾",
    layout="centered"
)

# Title
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image to predict the top 3 matching breeds.")

# Upload section (drag & drop + browse)
uploaded_file = st.file_uploader(
    "Drag & Drop or Browse Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily (since your function takes path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Predict
    with st.spinner("Analyzing... 🧠"):
        predictions = predict_top3(temp_path)

    st.subheader("🔍 Top 3 Predictions")

    # Display results
    for i, pred in enumerate(predictions, 1):
        st.write(f"**{i}. {pred['breed']}**")
        st.progress(pred["confidence"] / 100)
        st.write(f"{pred['confidence']}% confidence")
        st.write("---")

else:
    st.info("Please upload a dog image to get predictions.")