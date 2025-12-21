
<p align="center">
  <img src="https://svg-banners.vercel.app/api?type=origin&text1=ASL%20Recognition&width=900&height=250&color=blue" />
</p>

<h1 align="center" style="color:#3498DB;">
   ✋🏻 American Sign Language Recognition Project
</h1>

<div align="center">
  
  <img src="https://img.shields.io/badge/AI%20Project-ASL%20Recognition-3498DB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-1ABC9C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML%20Pipeline-Complete-9B59B6?style=for-the-badge" />

</div>

---

## 📝 *Project Description*

This project focuses on building an AI-powered system that recognizes *American Sign Language (ASL)* fingerspelling letters (A-Z) using machine-learning and deep-learning techniques.

The system learns from images of hand gestures and predicts the correct ASL letter based on:

✨ Finger position  
✨ Hand shape  
✨ Gesture orientation  

The project includes:

- 📁 Dataset preprocessing (cleaning, resizing, splitting)  
- 🧠 Training a CNN deep-learning model  
- 📈 Evaluating accuracy, loss & metrics  
- 🖥 Creating a friendly GUI for predictions  
- 📂 A fully organized and documented GitHub repository  

This results in a *complete ASL recognition tool* that can help translate ASL letters using AI.

---

## 🧠 Models Used

This project applies transfer learning using three different Convolutional Neural Network (CNN) architectures to recognize ASL fingerspelling letters:

- **ResNet50**
- **EfficientNetB0**
- **InceptionV3**

Each model was fine-tuned on the ASL Alphabet dataset and evaluated independently.  
A comparative analysis was conducted to determine the best-performing architecture based on accuracy, precision, and recall.

---

## 🔍 Model Explainability (Grad-CAM)

To enhance model interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to visualize the regions of the hand images that most influenced the model’s predictions.

Grad-CAM heatmaps highlight the important hand and finger regions used by the model during classification, ensuring that predictions are based on relevant visual features rather than background noise.

This explainability step improves transparency, trust, and reliability of the AI system.

---

## 🌿 *Main Deliverables*
- Clean preprocessed dataset 🧹  
- Trained CNN model 🤖  
- GUI for real-time letter prediction 🎨  
- Visualizations (accuracy, loss, confusion matrix) 📊  
- Full documentation 📚  

---

## 📂 *Repository Structure*

data/

models/

gui/

docs/

README.md

requirements.txt

.gitignore


---

## 🚀 *Features*
- Clean & well-processed dataset  
- Machine Learning training pipeline  
- Visualization graphs (accuracy, loss curves)  
- Model evaluation metrics  
- GUI for user-friendly interaction  
- Organized GitHub using branches + commits  
- Clear documentation  

---

## 🛠 *Technologies & Libraries Used*

| Category | Tools |
|---------|-------|
| *Programming Language* | Python 3.x |
| *AI / DL Libraries* | scikit-learn, TensorFlow  |
| *Data Handling* | pandas, numpy |
| *Visualization* | matplotlib, seaborn |
| *GUI* |  Streamlit |
| *Utilities* | joblib, pickle |



---

## 📊 Model Results & Comparison

The following table summarizes the performance of the three CNN architectures used in this project:

| Model | Accuracy | Precision | Recall |
|------|----------|-----------|--------|
| ResNet50 | XX% | XX% | XX% |
| EfficientNetB0 | XX% | XX% | XX% |
| InceptionV3 | XX% | XX% | XX% |

Based on the evaluation results, the best-performing model was selected for deployment in the GUI application.


---

### ▶ How to Run the Project

1. Install the required dependencies:
``bash
pip install -r requirements.txt

 2. python models/train.py

---

## 👥 *Team Roles*
| Member | Role | 
|--------|------|
| *Student 1* | Data preprocessing + documentation |
| *Student 2* | Model training + experiments |
| *Student 3* | GUI development |
| *Student 4* | GitHub maintenance + visualization |
| *Student 5* | *************************|
| *Student 6* | ********************|



---



## 📁 Dataset Sources
The dataset used in this project is the ASL Fingerspelling dataset from Kaggle:

" https://www.kaggle.com/datasets/dorukdemirci/asl-alphabet-dataset/data "


---



 
