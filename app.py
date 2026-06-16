import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="X-ray Disease Classifier", layout="centered")

st.title("🫁 Chest X-ray Disease Classification System")
st.write("Upload a chest X-ray image to predict: Covid / Normal / Viral Pneumonia")

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# LOAD MODEL (RESNET18 - MUST MATCH TRAINING)
# =========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

classes = ["Covid", "Normal", "Viral Pneumonia"]

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# GRAD-CAM FUNCTION
# =========================
def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.layer4[1].conv2

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()

    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.numpy()

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    # =========================
    # PREDICTION
    # =========================
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # =========================
    # RESULT
    # =========================
    st.subheader("🔍 Prediction Result")
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    # =========================
    # PROBABILITY CHART
    # =========================
    st.subheader("📊 Class Probabilities")

    prob_values = probs[0].numpy()

    fig, ax = plt.subplots()
    ax.bar(classes, prob_values)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Distribution")

    st.pyplot(fig)

    # =========================
    # GRAD-CAM VISUALIZATION
    # =========================
    st.subheader("🔥 Grad-CAM Heatmap (Model Focus Area)")

    heatmap = generate_gradcam(model, img_tensor)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)