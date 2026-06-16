
---

# 📄 FINAL PROJECT REPORT (DETAILED)

```md
# 🫁 FINAL PROJECT REPORT  
## Chest X-ray Disease Classification using Deep Learning

---

## 1. Introduction

Chest diseases such as COVID-19 and pneumonia require early detection for effective treatment.  
Radiology-based diagnosis is time-consuming and depends heavily on expert interpretation.

This project introduces a deep learning-based system that automatically classifies chest X-ray images into disease categories.

---

## 2. Objective

- To build an AI model for chest X-ray classification
- To improve diagnosis speed using deep learning
- To deploy the model using an interactive web application
- To provide explainability using Grad-CAM

---

## 3. Methodology

### 3.1 Data Collection
Chest X-ray images were collected and categorized into:
- COVID-19
- Normal
- Viral Pneumonia

---

### 3.2 Data Preprocessing
- Image resizing to 224×224
- Conversion to tensors
- Normalization for CNN input compatibility

---

### 3.3 Model Selection
A pretrained **ResNet18** model was used due to:
- Strong feature extraction capability
- Residual learning (solves vanishing gradient problem)
- Good performance on image classification tasks

Final layer was modified for 3-class output.

---

### 3.4 Training Process
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Training performed on labeled dataset
- Model saved as `model.pth`

---

## 4. System Architecture

Input Image → Preprocessing → ResNet18 Model → Softmax Output → Prediction

Additionally:
- Grad-CAM used for visualization of important regions

---

## 5. Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 6. Results

### Classification Report:

- Overall Accuracy: **85%**

| Class              | Precision | Recall | F1-score |
|-------------------|----------|--------|----------|
| Covid             | 0.92     | 0.92   | 0.92     |
| Normal            | 0.89     | 0.80   | 0.84     |
| Viral Pneumonia   | 0.73     | 0.80   | 0.76     |

---

## 7. Key Observations

- COVID detection performance is high and reliable
- Some confusion exists between Normal and Viral Pneumonia
- Model performs well but can improve with more data

---

## 8. Deployment

The trained model is deployed using **Streamlit**, allowing:
- Image upload
- Real-time prediction
- Confidence score display
- Grad-CAM visualization

---

## 9. Conclusion

This project demonstrates the effectiveness of deep learning in medical image classification.  
The system successfully assists in identifying chest diseases with good accuracy and interpretability.

---

## 10. Future Scope

- Improve accuracy using larger datasets
- Use advanced architectures (EfficientNet, DenseNet)
- Deploy on cloud platforms for public access
- Integrate with hospital systems