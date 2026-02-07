# Forensic Face Sketch Matching System Using Siamese Neural Network and Edge-based Feature Extraction 

A deep learning-based system designed to assist criminal investigations by automatically matching hand-drawn forensic sketches with real photographs using Siamese Neural Networks and Edge-Based Feature Extraction.

## 📌 Project Overview
Forensic investigations often rely on hand-drawn sketches to identify suspects. However, matching these sketches to real police databases is difficult due to the "modality gap"—sketches lack the texture, color, and lighting details of photos.

This project bridges that gap using a **Siamese Neural Network (SNN)** combined with **Edge-Based Feature Extraction**. Instead of traditional classification, the model learns a "similarity metric" to determine if a sketch and a photo belong to the same person.

## 🚀 Key Features
* **Cross-Modal Matching:** Successfully matches grayscale sketches to RGB photos.
* **Edge-Based Feature Extraction:** Uses Canny/Sobel edge detection to focus on facial contours rather than lighting or color.
* **Siamese Architecture:** Two identical CNN branches share weights to generate embeddings for comparison.
* **Similarity Scoring:** Outputs a precise distance score (Euclidean/Cosine) to rank potential matches.
* **Robustness:** Reduces the impact of lighting variations and noise.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV (cv2)
* **Data Handling:** NumPy, Pandas, Matplotlib
* **Interface (Optional):** Flask for web deployment

## 🏗️ System Architecture
The system follows this pipeline:
1.  **Input:** User uploads a Sketch and a Candidate Photo.
2.  **Preprocessing:** Images are resized, normalized, and converted to grayscale.
3.  **Edge Extraction:** Structural lines are extracted to align the modalities.
4.  **Siamese Network:** Both images pass through shared CNN layers to create 128-D embeddings.
5.  **Comparison:** The system calculates the distance between embeddings.
    * *Low Distance* = Match
    * *High Distance* = No Match

## 📊 Performance
* **Accuracy:** ~85–92% (depending on dataset quality).
* **False Non-Match Rate (FNMR):** Significantly reduced using edge features.
* **Inference Time:** Fast enough for real-time forensic usage.

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* TensorFlow
* OpenCV

### 1. Clone the Repository
```
git clone [](https://github.com/Gopika-9266/Forensic-Face-Sketch-Matching.git)

cd Forensic-Face-Sketch-Matching
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Run the Training Script 
```
python train_model.py
```
### 4. Test with an Image Pair
```
python match_faces.py --sketch "data/test_sketch.jpg" --photo "data/test_photo.jpg"
```
## 🔮 Future Scope
* GAN Integration: Using Generative Adversarial Networks to synthesize realistic photos from sketches.
* Real-Time Matching: Optimizing for large-scale police databases.
* Aging & Expression: Improving robustness against age progression.
