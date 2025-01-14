
import torch

class Config:
    # Device configuration
    DEVICE = 'cpu'  # Force CPU usage
    
    # Dataset parameters
    BATCH_SIZE = 32  # Reduced batch size for CPU
    NUM_WORKERS = 0  # Set to 0 for CPU
    
    # Model parameters
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Training parameters
    NUM_EPOCHS = 10  # Reduced epochs for CPU training
    
    # Paths
    MODEL_SAVE_PATH = 'model/cifar10_model.pth'
    LOG_DIR = 'logs/'
    
    # Data augmentation parameters
    RESIZE_SIZE = 128  # Reduced size for CPU
    CROP_SIZE = 128
    
    # Classes
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')