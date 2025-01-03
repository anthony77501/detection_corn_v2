import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

# Utilisation du CPU
device = torch.device('cpu')
print(f"Utilisation de l'appareil : {device}")

# Transformations pour les données
transform = transforms.Compose([
    transforms.Resize((128, 128)),      # Taille réduite pour les images
    transforms.RandomHorizontalFlip(), # Augmentation légère des données
    transforms.ToTensor(),             # Conversion en tenseur
])

# Chargement des données
train_data = datasets.ImageFolder('data/train', transform=transform)
val_data = datasets.ImageFolder('data/val', transform=transform)

# DataLoaders avec batch size réduit
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Affichage des classes
print(f"Classes détectées : {train_data.classes}")

# Charger un modèle pré-entraîné léger (MobileNetV2)
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes : Mûr et Non mûr
model = model.to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
epochs = 10
for epoch in range(epochs):
    # Mode entraînement
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Mode validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Résultats de l'époque
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'modele_de_mais.pth')
print("Modèle entraîné et sauvegardé sous 'modele_de_mais.pth'")
