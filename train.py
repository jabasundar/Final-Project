import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# TRANSFORMS (AUGMENTATION)
# ------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------------------
# DATASET
# ------------------------
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
test_data = datasets.ImageFolder("dataset/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# ------------------------
# MODEL
# ------------------------
model = get_model(3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------------
# TRAINING LOOP
# ------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ------------------------
# SAVE MODEL
# ------------------------
torch.save(model.state_dict(), "model.pth")
print("Model saved!")