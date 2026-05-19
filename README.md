# CIFAR-10 Image Classifier

A real-time image classification system using Custom CNN and Transfer Learning (MobileNetV2), trained on CIFAR-10 dataset and deployed as a web application using Streamlit.

---

## Problem Statement

Given a colour image, predict which of 10 predefined categories it belongs to. This is a multi-class image classification problem — the same technology used in self-driving cars, medical imaging, Google Lens, and content moderation systems.


## Dataset — CIFAR-10

Full Name : Canadian Institute for Advanced Research — 10 classes
Total Images : 60,000 (50,000 train / 10,000 test)
Image Size : 32 x 32 pixels, RGB colour
Classes : 10 — perfectly balanced at 6,000 images per class
Source : Loaded automatically via keras.datasets.cifar10

---

## The 10 Categories

Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck

---

## Project Files

```
image-classifier/
├── image_classifier.py    <- run this first to train the model
├── app/
│   └── streamlit_app.py   <- run this to launch the web app
├── models/                <- trained model saves here automatically
├── requirements.txt       <- list of libraries needed
├── .gitignore
└── README.md
```

---

## How to Run

### Step 1 - Install libraries
```
pip install -r requirements.txt
```

### Step 2 - Train the model
Open `image_classifier.py` in Google Colab and run all cells.

### Step 3 - Download the model
After training finishes, download `models/best_model.h5` from Colab.
Put it in the `models/` folder on your computer.

### Step 4 - Run the web app
```
streamlit run app/streamlit_app.py
```

Open your browser at http://localhost:8501 and upload any image!

---

## Results

Custom CNN : 75-76%
Transfer Learning (MobileNetV2) : 78-80%

---

## Key Observations:

Automobile and ship score highest — distinctive shapes even at 32x32
Cat and dog are the hardest to classify — they share similar visual features at low resolution
This cat/dog confusion is a known CIFAR-10 challenge, not a model flaw

---

## Web App Features

Image upload — supports JPG, PNG, BMP, WEBP
Real-time prediction — instant classification after upload
Confidence scores — interactive Plotly bar chart for all 10 classes
Top 3 predictions — ranked with progress bars
Model info — sidebar shows model type and input size automatically

---

## Tech Stack

- Python 3.10+
- TensorFlow / Keras (model training)
- Numpy
- Matplotlib / Seaborn
- Pandas
- Streamlit (web app)
- Plotly (charts)
- Pillow (image processing)
- scikit-learn (evaluation metrics)