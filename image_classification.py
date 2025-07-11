# from icrawler.builtin import BingImageCrawler

# def download_images(query, folder, num_images=200):
#     crawler = BingImageCrawler(storage={'root_dir': f'brainrot_dataset/{folder}'})
#     crawler.crawl(keyword=query, max_num=num_images)

# # Italian meme classes
# classes = {
#     "ciao bella meme": "ciao_bella",
#     "mamma mia meme": "mamma_mia",
#     "spaghetti moment meme": "spaghetti_moment",
#     "italian rage meme": "italian_rage",
#     "pizza cope meme": "pizza_cope",
#     "luigi stare meme": "luigi_stare"
# }

# for query, folder in classes.items():
#     print(f"Downloading {query}...")
#     download_images(query, folder)

import os
import shutil
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    random.seed(seed)

    # First create destination folders
    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            os.makedirs(os.path.join(dest_dir, split, class_name), exist_ok=True)

    # Now split using sklearn
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Step 1: train vs (val + test)
        train_files, temp_files = train_test_split(images, train_size=train_ratio, random_state=seed)

        # Step 2: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # proportion out of remaining
        val_files, test_files = train_test_split(temp_files, train_size=val_ratio_adjusted, random_state=seed)

        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split, split_images in splits.items():
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, split, class_name, img)
                shutil.copyfile(src, dst)

        print(f"✅ {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    print("\n🎉 Dataset splitting complete")


# 🧠 YOUR PATHS
source_dir = 'brainrot_dataset'
dest_dir = 'brainrot_dataset_split'

split_dataset(source_dir, dest_dir)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# Dataset paths
data_dir = 'brainrot_dataset_split'

# Transforms
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # VGG expects 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet mean
                             [0.229, 0.224, 0.225])  # Imagenet std
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform[x])
    for x in ['train', 'val', 'test']
}

# Dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in ['train', 'val', 'test']
}

class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VGG16 model
vgg_model = models.vgg16(pretrained=True)

# Freeze all layers
for param in vgg_model.parameters():
    param.requires_grad = False

# Replace the classifier (last layer)
num_classes = len(class_names)
vgg_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

vgg_model = vgg_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.classifier[6].parameters(), lr=0.001)

def train_model_vgg(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return model

# 🔥 Train VGG model
# vgg_model = train_model_vgg(vgg_model, dataloaders, criterion, optimizer, num_epochs=10)

# torch.save(vgg_model.state_dict(), "vgg16_brainrot.pth")
print("✅ Model saved successfully.")

# Set up model
vgg_model = models.vgg16(pretrained=True)
for param in vgg_model.parameters():
    param.requires_grad = False
vgg_model.classifier[6] = nn.Linear(in_features=4096, out_features=6) 

# Load weights
vgg_model.load_state_dict(torch.load("vgg16_brainrot.pth"))
vgg_model.eval()
vgg_model = vgg_model.to("cuda" if torch.cuda.is_available() else "cpu")

print("✅ Model loaded and ready to evaluate.")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def get_alexnet_model(num_classes):
    alexnet = models.alexnet(pretrained=True)

    # Freeze convolutional layers
    for param in alexnet.features.parameters():
        param.requires_grad = False

    # Replace classifier
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)

    return alexnet.to(device)


def load_alexnet_model(model_path="alexnet_brainrot.pth"):
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def train_model_alexnet(model, dataloaders, criterion, optimizer, num_epochs=10):
    print("\n🔧 Training AlexNet...")
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"\n✅ AlexNet Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model


def get_resnet18_model(num_classes):
    resnet = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace final FC layer
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    return resnet.to(device)

def train_model_resnet(model, dataloaders, criterion, optimizer, num_epochs=10):
    print("\n🔧 Training ResNet18...")
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"\n✅ ResNet18 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model


# 👇 Use one of these based on what you're testing
alexnet_model = get_alexnet_model(len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet_model.parameters(), lr=0.001)
# alexnet_trained_model = train_model_alexnet(alexnet_model, dataloaders, criterion, optimizer, num_epochs=10)
# Save the model
# torch.save(alexnet_trained_model.state_dict(), "alexnet_brainrot.pth")  

# Train ResNet
resnet_model = get_resnet18_model(len(class_names))
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
# resnet_trained_model = train_model_resnet(resnet_model, dataloaders, criterion, optimizer, num_epochs=10)
# torch.save(resnet_trained_model.state_dict(), "resnet18_brainrot.pth")

from sklearn.metrics import accuracy_score

# Run evaluation
true_labels, pred_labels = evaluate_model(vgg_model, dataloaders['test'])

# Print classification report
print("\n📊 Classification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))

# Optional: Print confusion matrix
print("🧩 Confusion Matrix:\n")
print(confusion_matrix(true_labels, pred_labels))

from PIL import Image
from torchvision import transforms

# Reuse the same transform used during validation/testing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path, model, class_names):
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    predicted_class = class_names[predicted.item()]
    print(f"🧠 Predicted class: *{predicted_class}*")

    return predicted_class

from glob import glob

test_dir = 'brainrot_dataset_split/test'
image_paths = glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True) + \
              glob(os.path.join(test_dir, '**', '*.jpeg'), recursive=True) + \
              glob(os.path.join(test_dir, '**', '*.png'), recursive=True)

# for img_path in image_paths:
#     print(f"\n🖼️ {os.path.basename(img_path)}")
#     predict_image(img_path, vgg_model, class_names)

results = {
    'Model': [],
    'Accuracy': [],
}

from torchvision.models import VGG16_Weights, AlexNet_Weights, ResNet18_Weights

for name, model_path in {
    "VGG16": "vgg16_brainrot.pth",
    "AlexNet": "alexnet_brainrot.pth",
    "ResNet18": "resnet18_brainrot.pth"
}.items():
    model = {
        "VGG16": lambda: models.vgg16(weights=VGG16_Weights.DEFAULT),
        "AlexNet": lambda: models.alexnet(weights=AlexNet_Weights.DEFAULT),
        "ResNet18": lambda: models.resnet18(weights=ResNet18_Weights.DEFAULT)
    }[name]()

    if name == "ResNet18":
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    else:
        model.classifier[6] = nn.Linear(4096, len(class_names))

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    y_true, y_pred = evaluate_model(model, dataloaders['test'])
    acc = accuracy_score(y_true, y_pred)

    results['Model'].append(name)
    results['Accuracy'].append(round(acc, 4))


# Display the comparison table
print(pd.DataFrame(results))





