# ================================================================
# streamlit_app.py — CIFAR-10 Image Classifier Web App
# ================================================================
# HOW TO RUN:
#   streamlit run app/streamlit_app.py
#
# MAKE SURE BEFORE RUNNING:
#   1. Training in Colab is complete
#   2. Download models/best_model.h5 from Colab
#   3. Put it inside your local models/ folder
# ================================================================

# --- Import libraries ---
import streamlit as st              # builds the entire web page
import numpy as np                  # works with numbers and arrays (same as notebook)
from PIL import Image               # opens image files uploaded by user
from tensorflow.keras.models import load_model   # loads our trained model
import plotly.graph_objects as go   # makes the interactive confidence bar chart
import os                           # checks if files exist on disk


# ================================================================
# CONFIGURATION
# ================================================================

# The exact same class names from our notebook.
class_names = [
    'airplane',    # label 0
    'automobile',  # label 1
    'bird',        # label 2
    'cat',         # label 3
    'deer',        # label 4
    'dog',         # label 5
    'frog',        # label 6
    'horse',       # label 7
    'ship',        # label 8
    'truck'        # label 9
]

# One emoji for each class — same order as class_names above
class_emojis = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐶', '🐸', '🐴', '🚢', '🚚']

# Path to our best saved model
# In notebook: shutil.copy('models/transfer_model.h5', 'models/best_model.h5')
# OR:          shutil.copy('models/custom_cnn.h5', 'models/best_model.h5')
MODEL_PATH = 'models/best_model.h5'


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔍",
    layout="wide"
)


# ================================================================
# LOAD THE MODEL
# ================================================================

@st.cache_resource
def load_my_model():
    """
    Load the trained model from disk.

    @st.cache_resource means this only runs ONCE when the app starts.
    Without this the model reloads every time someone uploads an image
    which would be very slow.

    This loads models/best_model.h5 — the same file saved by our notebook.
    The notebook saves whichever model performed better:
      - Custom CNN           -> saved as models/custom_cnn.h5 (32x32 input)
      - Transfer Learning    -> saved as models/transfer_model.h5 (64x64 input)
      - The better one is copied to models/best_model.h5 for the app to use
    """
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found at 'models/best_model.h5'")
        st.info(
            "Please run the Jupyter notebook in Colab first, "
            "then download best_model.h5 and put it in the models/ folder."
        )
        st.stop()

    return load_model(MODEL_PATH)


# Load the model — runs once when app starts
model = load_my_model()

# Detect what image size the model expects
# Custom CNN:         input_shape = (None, 32, 32, 3)
# Transfer Learning:  input_shape = (None, 64, 64, 3)
# This matches how we checked in notebook Cell 38:
#   input_shape = model.input_shape
#   if input_shape[1] == 64: use X_test_64 else use X_test
input_h = model.input_shape[1]
input_w = model.input_shape[2]


# ================================================================
# HELPER FUNCTION — Prepare Image for the Model
# ================================================================

def prepare_image(uploaded_image):
    """
    Prepares an uploaded image so the model can make a prediction.

    Does the same preprocessing steps as in our notebook:
      Step 1 - Convert to RGB (handles PNG with transparency, greyscale etc.)
      Step 2 - Resize to correct size (32x32 or 64x64 depending on model)
               Notebook: X_train.shape = (10000, 32, 32, 3)
      Step 3 - Normalise pixels from 0-255 to 0.0-1.0
               Notebook: X_train = X_train.astype('float32') / 255.0
      Step 4 - Add batch dimension so shape is (1, H, W, 3)
               Model always expects a batch, not a single image

    Args:
        uploaded_image: PIL Image opened from the file uploader

    Returns:
        numpy array of shape (1, H, W, 3) ready for model.predict()
    """
    img      = uploaded_image.convert("RGB")               # ensure 3 colour channels
    img      = img.resize((input_w, input_h))               # resize to model input size
    img_arr  = np.array(img).astype('float32') / 255.0     # normalise pixels to 0-1
    img_arr  = np.expand_dims(img_arr, axis=0)              # add batch dim -> (1, H, W, 3)
    return img_arr


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.title("About This App")
    st.markdown("---")

    st.markdown("""
This web app uses a deep learning model trained on the
**CIFAR-10 dataset** to classify images into 10 categories.

Upload any image and the AI will tell you what it thinks it is,
along with a confidence score for all 10 classes.
""")

    st.markdown("**Model Info**")
    st.markdown(f"- Input size  : {input_h} x {input_w} pixels")
    st.markdown(
        f"- Model type  : "
        f"{'Transfer Learning (MobileNetV2)' if input_h == 64 else 'Custom CNN'}"
    )
    st.markdown("- Dataset     : CIFAR-10")
    st.markdown("- Classes     : 10")

    st.markdown("---")
    st.markdown("**10 Supported Classes**")

    for emoji, name in zip(class_emojis, class_names):
        st.write(f"{emoji} {name.capitalize()}")

    st.markdown("---")
    st.markdown("""
**Tips for best results:**
- Use images with one clear subject
- Simple backgrounds work best
- Subject should match one of the 10 classes above
""")


# ================================================================
# MAIN PAGE
# ================================================================

st.title("CIFAR-10 Real-Time Image Classifier")
st.markdown(
    "Upload any image and the AI will classify it into one of "
    "**10 categories** with a confidence score."
)
st.markdown("---")

# Two column layout — upload on left, results on right
left_col, right_col = st.columns(2, gap="large")


# ---- LEFT COLUMN: Image Upload ----
with left_col:
    st.subheader("Step 1 — Upload Your Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a photo of any of the 10 CIFAR-10 classes"
    )

    if uploaded_file is not None:
        # Open uploaded file as PIL Image
        image = Image.open(uploaded_file).convert("RGB")

        # Show the image on screen
        st.image(image, caption="Your uploaded image", use_container_width=True)

        # Show size info
        st.caption(f"Original size : {image.size[0]} x {image.size[1]} pixels")
        st.caption(f"Resized to    : {input_h} x {input_w} pixels for the model")


# ---- RIGHT COLUMN: Prediction Results ----
with right_col:
    st.subheader("Step 2 — Prediction Results")

    if uploaded_file is not None:

        # Run preprocessing — same steps as notebook
        with st.spinner("Analysing image..."):
            prepared    = prepare_image(image)

            # Get predictions — same as notebook:
            # y_pred_prob = model.predict(X_test_eval, verbose=0)
            predictions = model.predict(prepared, verbose=0)[0]
            # predictions = array of 10 numbers, each = probability for that class
            # all 10 numbers add up to 1.0

        # Find the winning class — same as:
        # y_pred = np.argmax(y_pred_prob, axis=1)
        best_idx   = int(np.argmax(predictions))
        best_name  = class_names[best_idx]
        best_emoji = class_emojis[best_idx]
        confidence = float(predictions[best_idx]) * 100

        # Show the prediction
        st.success(f"**Prediction: {best_emoji} {best_name.upper()}**")
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.1f}%",
            help="How confident the model is in this prediction"
        )

        st.markdown("---")

        # Bar chart showing confidence for all 10 classes
        # This visually shows the same info as the classification report in the notebook
        st.subheader("Confidence for All 10 Classes")

        labels = [f"{class_emojis[i]} {class_names[i]}" for i in range(10)]
        values = [round(float(p) * 100, 1) for p in predictions]

        # Predicted class gets dark blue, others get light blue
        bar_colors = [
            '#1a73e8' if i == best_idx else '#aecde8'
            for i in range(10)
        ]

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=bar_colors,
            text=[f"{v}%" for v in values],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 115]),
            height=380,
            margin=dict(l=10, r=10, t=10, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Top 3 predictions with progress bars
        st.subheader("🏆 Top 3 Predictions")

        # Sort all predictions highest to lowest, take top 3
        top3_indices = np.argsort(predictions)[::-1][:3]
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]

        for medal, idx in zip(medals, top3_indices):
            name  = class_names[idx]
            emoji = class_emojis[idx]
            pct   = float(predictions[idx]) * 100
            st.write(f"{medal}: **{emoji} {name}** — {pct:.1f}%")
            st.progress(int(pct))

    else:
        # Placeholder shown when no image uploaded yet
        st.info("👆 Upload an image on the left to see the prediction here!")
        st.markdown("**This model can recognise:**")
        col1, col2 = st.columns(2)
        for i, (emoji, name) in enumerate(zip(class_emojis, class_names)):
            if i < 5:
                col1.write(f"{emoji} {name.capitalize()}")
            else:
                col2.write(f"{emoji} {name.capitalize()}")


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")
st.caption(
    "Built with TensorFlow · Keras · Streamlit · Plotly  |  "
    "Dataset: CIFAR-10  |  "
    "Model: Custom CNN + MobileNetV2 Transfer Learning"
)