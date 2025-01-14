import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import get_transforms, plot_metrics, plot_confusion_matrix
from config import Config
import os

def train_model():
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=Config.BATCH_SIZE, 
        shuffle=True, num_workers=Config.NUM_WORKERS
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    # Initialize model with proper weights parameter
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    model = model.to(device)
    
    # Rest of the training code remains the same
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(testloader)
        val_acc = 100. * correct / total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    
    # Plot metrics
    plot_metrics(history)
    plot_confusion_matrix(model, testloader, device)

if __name__ == "__main__":
    train_model()