# ================================================================
# streamlit_app.py — CIFAR-10 Image Classifier + YOLOv8 Object Detection
# ================================================================
# HOW TO RUN:
#   streamlit run app/streamlit_app.py
#
# MAKE SURE BEFORE RUNNING:
#   1. Training in Colab is complete
#   2. Download models/best_model.h5 from Colab
#   3. Put it inside your local models/ folder
#   4. detect.py must be in the ROOT image-classifier/ folder
# ================================================================

# --- Import libraries ---
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
import sys
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add root folder to path so we can import detect.py from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detect import detect_objects


# ================================================================
# CONFIGURATION
# ================================================================

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class_emojis = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐶', '🐸', '🐴', '🚢', '🚚']

MODEL_PATH = 'models/best_model.h5'


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Image AI System",
    page_icon="🔍",
    layout="wide"
)


# ================================================================
# LOAD THE CLASSIFIER MODEL
# ================================================================

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found at 'models/best_model.h5'")
        st.info(
            "Please run the Jupyter notebook in Colab first, "
            "then download best_model.h5 and put it in the models/ folder."
        )
        st.stop()
    return load_model(MODEL_PATH)


model = load_my_model()

input_h = model.input_shape[1]
input_w = model.input_shape[2]


# ================================================================
# HELPER FUNCTION — Prepare Image for Classifier
# ================================================================

def prepare_image(uploaded_image):
    img     = uploaded_image.convert("RGB")
    img     = img.resize((input_w, input_h))
    img_arr = np.array(img).astype('float32') / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


# ================================================================
# GRAD-CAM — Model Explainability
# ================================================================
# Shows which part of the image the model focused on when making
# its prediction. Targets an earlier internal MobileNetV2 layer
# (block_13_expand_relu) rather than the final layer, since the
# model's 64x64 input leaves only a 2x2 feature map at the very end
# — too coarse to localize anything meaningfully. The earlier layer
# gives a 4x4 grid, which produces genuinely useful heatmaps.

@st.cache_resource
def build_gradcam_extractor(_model):
    """
    Locates the nested MobileNetV2 submodel inside the loaded model and
    builds a small feature-extractor model targeting an earlier internal
    layer for better spatial resolution. Returns None if the loaded model
    isn't a transfer-learning model (e.g. if best_model.h5 turned out to
    be the custom CNN instead) — Grad-CAM is skipped gracefully in that case.
    """
    nested_base = None
    base_index = None
    for i, layer in enumerate(_model.layers):
        if isinstance(layer, tf.keras.Model):
            nested_base = layer
            base_index = i
            break

    if nested_base is None:
        return None

    layers_after_base = _model.layers[base_index + 1:]
    target_layer_name = "block_13_expand_relu"

    try:
        feature_extractor = tf.keras.Model(
            inputs=nested_base.input,
            outputs=[nested_base.get_layer(target_layer_name).output, nested_base.output]
        )
    except ValueError:
        return None

    return {"feature_extractor": feature_extractor, "layers_after_base": layers_after_base}


def compute_gradcam(gradcam_bundle, prepared_image):
    """
    Runs Grad-CAM on a single prepared image (already resized/normalized,
    shape (1, H, W, 3)) and returns a 32x32 heatmap array in 0-1 range.
    """
    feature_extractor = gradcam_bundle["feature_extractor"]
    layers_after_base = gradcam_bundle["layers_after_base"]

    img = tf.cast(prepared_image, tf.float32)

    with tf.GradientTape() as tape:
        conv_output, base_out = feature_extractor(img, training=False)
        tape.watch(conv_output)
        x = base_out
        for layer in layers_after_base:
            x = layer(x, training=False)
        preds = x
        top_class = tf.argmax(preds[0])
        score = preds[:, top_class]

    grads = tape.gradient(score, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_output[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = np.array(
        tf.image.resize([heatmap[:, :, np.newaxis]], [32, 32])
    ).squeeze()

    return heatmap_resized


def make_overlay_figure(original_image, heatmap):
    """
    Returns a matplotlib figure with the original image, the raw heatmap,
    and the overlay side by side, styled for display in Streamlit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap\n(bright = focused here)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(original_image)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.title("About This App")
    st.markdown("---")
    st.markdown("""
This web app combines two computer vision systems:

**Tab 1 — Image Classification**  
Uses MobileNetV2 transfer learning trained on CIFAR-10 to classify 
images into 10 categories with confidence scores.

**Tab 2 — Object Detection**  
Uses YOLOv8 (pretrained on COCO dataset) to detect and spatially 
locate 80 types of objects in any uploaded image with bounding boxes.
""")

    st.markdown("**Classifier Model Info**")
    st.markdown(f"- Input size  : {input_h} x {input_w} pixels")
    st.markdown(
        f"- Model type  : "
        f"{'Transfer Learning (MobileNetV2)' if input_h == 64 else 'Custom CNN'}"
    )
    st.markdown("- Dataset     : CIFAR-10")
    st.markdown("- Classes     : 10")

    st.markdown("---")
    st.markdown("**10 Classifier Classes**")
    for emoji, name in zip(class_emojis, class_names):
        st.write(f"{emoji} {name.capitalize()}")

    st.markdown("---")
    st.markdown("**YOLOv8 detects 80 objects including:**")
    st.write("👤 person · 🚗 car · 🐕 dog · 🐈 cat · 🚌 bus")
    st.write("🚲 bicycle · ✈️ airplane · 🪑 chair · 💻 laptop · 📱 phone")

    st.markdown("---")
    st.markdown("""
**Tips for best results:**
- Classification: use images matching the 10 CIFAR classes
- Detection: use any real-world photo with people or objects
- Simple backgrounds work best for classification
""")


# ================================================================
# MAIN PAGE — TITLE
# ================================================================

st.title("🔍 Image AI System")
st.markdown("**Image Classification** (CIFAR-10 · MobileNetV2)  &  **Object Detection** (YOLOv8)")
st.markdown("---")


# ================================================================
# TABS
# ================================================================

tab1, tab2 = st.tabs([
    "🏷️  Image Classification (CIFAR-10)",
    "📦  Object Detection (YOLOv8)"
])


# ================================================================
# TAB 1 — IMAGE CLASSIFICATION
# ================================================================

with tab1:
    st.markdown("Upload any image and the AI will classify it into one of **10 categories** with a confidence score.")
    st.markdown("---")

    left_col, right_col = st.columns(2, gap="large")

    # LEFT: Upload
    with left_col:
        st.subheader("Step 1 — Upload Your Image")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a photo of any of the 10 CIFAR-10 classes",
            key="classifier_uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded image", use_container_width=True)
            st.caption(f"Original size : {image.size[0]} x {image.size[1]} pixels")
            st.caption(f"Resized to    : {input_h} x {input_w} pixels for the model")

    # RIGHT: Results
    with right_col:
        st.subheader("Step 2 — Prediction Results")

        if uploaded_file is not None:
            with st.spinner("Analysing image..."):
                prepared    = prepare_image(image)
                predictions = model.predict(prepared, verbose=0)[0]

            best_idx   = int(np.argmax(predictions))
            best_name  = class_names[best_idx]
            best_emoji = class_emojis[best_idx]
            confidence = float(predictions[best_idx]) * 100

            st.success(f"**Prediction: {best_emoji} {best_name.upper()}**")
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.1f}%",
                help="How confident the model is in this prediction"
            )

            st.markdown("---")
            st.subheader("Confidence for All 10 Classes")

            labels = [f"{class_emojis[i]} {class_names[i]}" for i in range(10)]
            values = [round(float(p) * 100, 1) for p in predictions]

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

            st.subheader("🏆 Top 3 Predictions")

            top3_indices = np.argsort(predictions)[::-1][:3]
            medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]

            for medal, idx in zip(medals, top3_indices):
                name  = class_names[idx]
                emoji = class_emojis[idx]
                pct   = float(predictions[idx]) * 100
                st.write(f"{medal}: **{emoji} {name}** — {pct:.1f}%")
                st.progress(int(pct))

            # ================================================================
            # GRAD-CAM SECTION — Why did the model predict this?
            # ================================================================
            st.markdown("---")
            st.subheader("🔬 Why did the model predict this?")
            st.caption(
                "Grad-CAM highlights the region of the image the model focused on "
                "most when making its prediction. Bright red/yellow = high influence."
            )

            gradcam_bundle = build_gradcam_extractor(model)

            if gradcam_bundle is None:
                st.info(
                    "Grad-CAM isn't available for this model's architecture "
                    "(expected a transfer-learning model with a nested base network)."
                )
            else:
                with st.spinner("Generating explanation..."):
                    heatmap = compute_gradcam(gradcam_bundle, prepared)
                    original_for_display = np.array(image.resize((32, 32)))
                    gradcam_fig = make_overlay_figure(original_for_display, heatmap)

                gcol1, gcol2 = st.columns([2, 1])
                with gcol1:
                    st.pyplot(gradcam_fig, use_container_width=True)
                with gcol2:
                    st.markdown("**How to read this:**")
                    st.markdown(
                        "- 🔴 **Red/orange** — strongest influence on the prediction\n"
                        "- 🔵 **Blue** — little to no influence\n\n"
                        "If the highlighted region doesn't line up with the actual "
                        "object, that's often a sign the model got confused by "
                        "background texture or shape — which can help explain "
                        "an incorrect prediction."
                    )
                plt.close(gradcam_fig)

        else:
            st.info("👆 Upload an image on the left to see the prediction here!")
            st.markdown("**This model can recognise:**")
            col1, col2 = st.columns(2)
            for i, (emoji, name) in enumerate(zip(class_emojis, class_names)):
                if i < 5:
                    col1.write(f"{emoji} {name.capitalize()}")
                else:
                    col2.write(f"{emoji} {name.capitalize()}")


# ================================================================
# TAB 2 — OBJECT DETECTION (YOLOv8)
# ================================================================

with tab2:
    st.markdown("Upload any real-world photo to **detect and locate objects** using YOLOv8.")
    st.markdown("Detects **80 object categories** including people, vehicles, animals, and everyday items.")
    st.markdown("---")

    det_left, det_right = st.columns(2, gap="large")

    # LEFT: Upload
    with det_left:
        st.subheader("Step 1 — Upload Your Image")

        detection_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload any photo — street scene, room, animals, people etc.",
            key="detection_uploader"
        )

        if detection_file is not None:
            det_image = Image.open(detection_file).convert("RGB")
            st.image(det_image, caption="Your uploaded image", use_container_width=True)
            st.caption(f"Image size: {det_image.size[0]} x {det_image.size[1]} pixels")

    # RIGHT: Detection Results
    with det_right:
        st.subheader("Step 2 — Detection Results")

        if detection_file is not None:
            with st.spinner("Detecting objects with YOLOv8..."):
                annotated_img, detections = detect_objects(det_image)

            if detections:
                st.image(
                    annotated_img,
                    caption=f"Detected {len(detections)} object(s) with bounding boxes",
                    use_container_width=True
                )
            else:
                st.image(det_image, caption="No objects detected", use_container_width=True)

            st.markdown("---")

            if detections:
                st.subheader(f"✅ Found {len(detections)} Object(s)")

                # Group by object type and show counts
                object_counts = {}
                for d in detections:
                    obj = d['object']
                    object_counts[obj] = object_counts.get(obj, 0) + 1

                # Summary row
                summary = " · ".join([
                    f"**{count}x {obj}**"
                    for obj, count in object_counts.items()
                ])
                st.markdown(summary)
                st.markdown("---")

                # Detailed list
                for i, d in enumerate(detections, 1):
                    conf_color = "🟢" if d['confidence'] >= 70 else "🟡" if d['confidence'] >= 50 else "🔴"
                    st.markdown(f"""
**{i}. {d['object'].upper()}** {conf_color}
- Confidence: `{d['confidence']}%`
- Location: `{d['location']}`
""")

                # Confidence bar chart for detections
                if len(detections) <= 20:
                    st.markdown("---")
                    st.subheader("Confidence Scores")

                    det_labels = [f"{d['object']} #{i+1}" for i, d in enumerate(detections)]
                    det_values = [d['confidence'] for d in detections]
                    det_colors = [
                        '#1a73e8' if v >= 70 else '#f4a942' if v >= 50 else '#e84242'
                        for v in det_values
                    ]

                    fig2 = go.Figure(go.Bar(
                        x=det_labels,
                        y=det_values,
                        marker_color=det_colors,
                        text=[f"{v}%" for v in det_values],
                        textposition='outside'
                    ))

                    fig2.update_layout(
                        xaxis_title="Detected Object",
                        yaxis_title="Confidence (%)",
                        yaxis=dict(range=[0, 115]),
                        height=350,
                        margin=dict(l=10, r=10, t=10, b=60),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig2, use_container_width=True)

            else:
                st.warning("No objects detected in this image.")
                st.markdown("""
**Try these tips:**
- Use a clearer, higher resolution photo
- Make sure objects are large enough in the frame
- YOLOv8 works best on real-world photos (not artwork or abstract images)
""")

        else:
            st.info("👆 Upload an image on the left to detect objects!")
            st.markdown("""
**What YOLOv8 can detect:**

| Category | Examples |
|---|---|
| People | person |
| Vehicles | car, truck, bus, bicycle, motorcycle |
| Animals | dog, cat, bird, horse, sheep, cow |
| Electronics | laptop, phone, TV, keyboard |
| Furniture | chair, sofa, bed, dining table |
| Kitchen | bottle, cup, fork, knife, bowl |
| Sports | sports ball, skateboard, tennis racket |
""")


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")
st.caption(
    "Built with TensorFlow · Keras · YOLOv8 · Ultralytics · Streamlit · Plotly  |  "
    "Classification: CIFAR-10 + MobileNetV2  |  "
    "Detection: YOLOv8n (COCO dataset · 80 classes)"
)