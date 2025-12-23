import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import io

# ✅ Streamlit page setup
st.set_page_config(page_title="3D Print Failure Classifier", layout="centered")
st.title("🖨️ 3D Printing Failure Detector with LIME Explanation")

# ✅ Model selection
model_options = {
    "Model 1 (Base)": "3d_printing_classifier.h5",
    "Model 2 (Variant A)": "3d_printing_classifier0.h5",
    "Model 3 (Variant B)": "3d_printing_classifier1.h5"
}

selected_model_name = st.selectbox("Select a Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# ✅ Load model based on selection
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

model = load_trained_model(selected_model_path)
st.success(f"{selected_model_name} loaded successfully!")

# ✅ Image uploader
uploaded_file = st.file_uploader("Upload a 3D print image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_resized = image_pil.resize((224, 224))
    img_array = image.img_to_array(image_resized)

    # Normalize and batch
    img_array_norm = img_array / 255.0
    img_batch = np.expand_dims(img_array_norm, axis=0)

    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Prediction
    prediction = model.predict(img_batch)[0][0]
    label = "✅ Success" if prediction > 0.5 else "❌ Failure"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.progress(float(confidence))
    st.markdown(f"**Model Confidence:** `{confidence:.2%}`")

    # LIME explainability
    if st.button("Explain with LIME"):
        with st.spinner("Explaining prediction using LIME..."):

            # Define LIME prediction function
            def predict_fn(images):
                images = np.array(images).astype('float32') / 255.0
                return model.predict(images)

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array.astype('double'),
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=100
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )

            superimposed_image = mark_boundaries(temp.astype(np.uint8), mask)

            fig, ax = plt.subplots()
            ax.imshow(superimposed_image)
            ax.set_title("LIME Explanation")
            ax.axis('off')
            st.pyplot(fig)

            # Optional download
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="📥 Download LIME Explanation",
                data=buf,
                file_name="lime_explanation.png",
                mime="image/png"
            )
