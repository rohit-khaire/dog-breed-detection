import streamlit as st
from PIL import Image
import tempfile
import cv2
from app import predict_multiple_dogs
from breed_api import get_breed_info

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Dog AI",
    page_icon="🐾",
    layout="wide"
)

# -----------------------------
# Custom Styling (Modern UI)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("🐶 Dog AI - Detection & Breed Intelligence")
st.caption("Upload an image to detect dogs and identify their breeds with AI")

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & Drop or Browse Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    # -----------------------------
    # LEFT: Uploaded Image
    # -----------------------------
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

    # -----------------------------
    # RIGHT: Processing
    # -----------------------------
    with col2:
        with st.spinner("🧠 Analyzing image..."):
            results = predict_multiple_dogs(temp_path)

        # -----------------------------
        # Detection Status
        # -----------------------------
        if results[0]["type"] == "fallback":
            st.warning("⚠️ No clear dog detected — showing best guess")
        else:
            st.success(f"✅ Detected {len(results)} dog(s)")

        # -----------------------------
        # Confidence Feedback
        # -----------------------------
        if results[0]["type"] == "fallback":
            conf = results[0]["predictions"][0]["confidence"]

            if conf < 20:
                st.error("❌ This image likely does NOT contain a dog")
            elif conf < 40:
                st.warning("⚠️ Low confidence prediction")
            else:
                st.info("ℹ️ Showing best guess")

    # -----------------------------
    # Draw Bounding Boxes
    # -----------------------------
    if results[0]["type"] != "fallback":
        image_cv = cv2.imread(temp_path)

        for res in results:
            x1, y1, x2, y2 = res["box"]
            top_pred = res["predictions"][0]

            label = f"{top_pred['breed']} ({top_pred['confidence']}%)"

            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB),
                 caption="🎯 Detected Dogs", use_container_width=True)

    # -----------------------------
    # Predictions Section (Cards UI)
    # -----------------------------
    st.markdown("---")
    st.subheader("🔍 Predictions")

    for i, res in enumerate(results, 1):
        if res["type"] == "fallback":
            st.write("### 🖼️ Full Image Prediction")
        else:
            st.write(f"### 🐶 Dog {i}")

        for j, pred in enumerate(res["predictions"]):
            breed_name = pred["breed"]

            with st.container():
                st.markdown(f"""
                <div style="background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:10px">
                    <h4 style="color:white">{breed_name}</h4>
                    <p style="color:#bbb">{pred['confidence']}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(pred["confidence"] / 100)

                # 🔥 UNIQUE BUTTON KEY
                btn_key = f"btn_{i}_{j}"

                if st.button(f"View More about {breed_name}", key=btn_key):

                    with st.spinner("Fetching breed info..."):
                        info = get_breed_info(breed_name)

                    # 🔥 SHOW INFO CARD
                    if "error" in info:
                        st.error("No detailed info available")
                    else:
                        st.markdown(f"""
                        <div style="background:#111;padding:20px;border-radius:15px;margin-top:10px">
                            <h3 style="color:#4CAF50">{info['name']}</h3>
                            <p style="color:white"><b>Origin:</b> {info['origin']}</p>
                            <p style="color:white"><b>Life Span:</b> {info['life_span']}</p>
                            <p style="color:white"><b>Weight:</b> {info['weight']}</p>
                            <p style="color:white"><b>Height:</b> {info['height']}</p>
                            <p style="color:white"><b>Temperament:</b> {info['temperament']}</p>
                            <p style="color:white"><b>Bred For:</b> {info['bred_for']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.write("---")

else:
    st.info("👆 Upload an image to get started")