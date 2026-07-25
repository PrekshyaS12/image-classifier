# Image AI System — CIFAR-10 Classifier + YOLOv8 Object Detection

A dual-capability computer vision web application combining image classification (MobileNetV2 transfer learning on CIFAR-10) and real-time object detection (YOLOv8), deployed as an interactive Streamlit web app with Grad-CAM explainability.

---

## What This System Does

This project combines two distinct computer vision capabilities in one application:

**Image Classification** — Given a colour image, predict which of 10 predefined CIFAR-10 categories it belongs to, with confidence scores for all classes. The same technology powers Google Lens, content moderation systems, and medical imaging tools.

**Object Detection** — Given any real-world photo, detect and spatially locate multiple objects simultaneously using bounding boxes. The same approach is used in autonomous vehicles, surveillance systems, and robotics perception pipelines.

**Explainability** — Every classification prediction comes with a Grad-CAM visualization showing exactly which part of the image the model focused on, so predictions (including incorrect ones) can actually be understood, not just trusted blindly.

---

## Dataset — CIFAR-10 (Classification)

| Property | Detail |
|---|---|
| Full Name | Canadian Institute for Advanced Research — 10 classes |
| Total Images | 60,000 (50,000 train / 10,000 test) |
| Image Size | 32 × 32 pixels, RGB colour |
| Classes | 10 — perfectly balanced at 6,000 images per class |
| Source | Loaded automatically via `keras.datasets.cifar10` |

### The 10 Categories
`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## Dataset — COCO (Object Detection)

YOLOv8 is pretrained on the COCO (Common Objects in Context) dataset — 80 object categories including people, vehicles, animals, furniture, electronics, kitchen items, and sports equipment. No additional training required.

---

## Project Structure

```
image-classifier/
├── detect.py                  ← YOLOv8 detection helper (root level)
├── test_detect.py             ← test script to verify YOLOv8 works
├── app/
│   └── streamlit_app.py       ← Streamlit web app (both tabs + Grad-CAM)
├── models/
│   └── best_model.h5          ← trained CIFAR-10 model
├── notebooks/
│   └── image_classifier.ipynb ← training notebook (Colab or local)
├── plots/
│   └── gradcam_*.png          ← saved Grad-CAM comparisons from the notebook
├── src/                       ← source utilities
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### Step 1 — Install libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Train the CIFAR-10 classifier
Open `notebooks/image_classifier.ipynb` — this can be run either:
- **In Google Colab** (original workflow, useful for free GPU access) — includes a Drive-mount cell at the start and a download cell at the end for pulling the trained model off Colab's cloud instance.
- **Locally** (e.g. VS Code / Jupyter) — skip the two Colab-specific cells (Drive mount and file download); everything else uses local relative paths and works as-is. `ModelCheckpoint` saves the trained model directly to `models/` during training, so no manual download step is needed locally.

### Step 3 — Confirm the trained model is present
After training, `models/best_model.h5` should exist on disk (either downloaded from Colab or saved automatically if trained locally).

### Step 4 — Launch the web app
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`

---

## Results

### CIFAR-10 Classification
| Model | Test Accuracy |
|---|---|
| Custom CNN | 75–76% |
| Transfer Learning (MobileNetV2) | 78–80% |

### Key Observations
- **Automobile and ship** score highest — distinctive shapes even at 32×32 resolution
- **Cat and dog** are hardest to classify — they share similar visual texture at low resolution
- MobileNetV2 pretrained weights provide a 3–5% accuracy gain over a custom CNN with the same training data

---

## Explainability — Grad-CAM

Every prediction — correct or incorrect — comes with a Grad-CAM heatmap showing which region of the image most influenced the model's decision.

**A key technical detail:** MobileNetV2 downsamples the image by a factor of 32 as it passes through the network. At the model's 64×64 input size, the *final* convolutional layer produces only a 2×2 feature map — too coarse to meaningfully localize anything. Instead, Grad-CAM here targets an earlier internal layer (`block_13_expand_relu`), which produces a 4×4 feature map and gives genuinely useful, spatially meaningful heatmaps.

**What this revealed:** on correct predictions, the heatmap consistently lands directly on the actual object (e.g. centered precisely on a bird's head, or a horse's body). On misclassifications, the heatmap often explains *why* the model got it wrong — for example, a bird photographed against diagonal tree branches was misclassified as "airplane," with the heatmap showing the model had fixated on the branch lines rather than the bird itself.

This is available live in the Streamlit app (Tab 1, under "Why did the model predict this?") for any uploaded image, and as saved comparison plots in `plots/` from the notebook.

### YOLOv8 Object Detection
- Detects 80 object categories in any uploaded image
- Returns bounding box coordinates and confidence scores per object
- YOLOv8n (nano) model — fast inference, ~6MB download, no GPU required for demo use

---

## Web App Features

### Tab 1 — Image Classification
- Image upload: JPG, PNG, BMP, WEBP
- Real-time prediction with confidence score
- Interactive Plotly bar chart for all 10 classes
- Top 3 predictions ranked with progress bars
- Grad-CAM visual explainability (shows which image regions drove the prediction, live for any uploaded image)
- Sidebar shows model type and input size automatically

### Tab 2 — Object Detection (YOLOv8)
- Image upload: any real-world photo
- Bounding box visualisation drawn directly on image
- Per-object confidence scores with colour coding (🟢 High · 🟡 Medium · 🔴 Low)
- Object count summary grouped by category
- Interactive Plotly bar chart of all detection confidence scores
- Detailed location coordinates for each detected object

---

## Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | TensorFlow / Keras, MobileNetV2 |
| Object Detection | YOLOv8 (Ultralytics), COCO pretrained weights |
| Explainability | Grad-CAM (gradient-weighted class activation mapping) |
| Web App | Streamlit |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Image Processing | Pillow (PIL) |
| Data | NumPy, Pandas |
| Evaluation | scikit-learn |
| Language | Python 3.10+ |

---

## Why Two Models?

Classification and detection solve different problems:

**Classification** answers: *"What is this image?"* — one label for the whole image.

**Detection** answers: *"What is in this image and where?"* — multiple labels with spatial coordinates.

Combining both in one system demonstrates how real-world AI pipelines layer perception capabilities — the same architecture used in autonomous robots that must first detect objects in a scene before deciding how to respond to them.