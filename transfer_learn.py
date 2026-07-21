import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np


# --- Data Loading (ResNet needs 224x224 input) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.EuroSAT(root="data", download=False, transform=transform)
class_names = dataset.classes

generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load pretrained ResNet18
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers except the final classifier
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace final layer for 10 EuroSAT classes
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    resnet = resnet.to(device)

    print("ResNet18 loaded with pretrained ImageNet weights")
    print(f"Final layer: Linear(in_features={resnet.fc.in_features}, out_features=10)")
    print("Only training the final classification layer\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
    
    # Train for 5 epochs (transfer learning converges fast)
    print("--- Training (Transfer Learning) ---")
    for epoch in range(5):
        resnet.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/5], Accuracy: {100*correct/total:.2f}%")

        # Evaluate
    print("\n--- Evaluation ---")
    resnet.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    resnet_accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)

    print(f"\nResNet18 Transfer Learning Test Accuracy: {resnet_accuracy:.2f}%")
    print("\nCompare this to your custom CNN's test accuracy!")
    print("Transfer learning leverages features learned from millions of ImageNet")
    print("images to achieve higher accuracy with less training time.")