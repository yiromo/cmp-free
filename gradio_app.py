import gradio as gr
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from config import Config

class CIFAR10Predictor:
    def __init__(self, model_path):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, Config.NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.RESIZE_SIZE, Config.RESIZE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict(self, image):
        # Convert Gradio image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Transform and predict
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        results = {Config.CLASSES[idx]: float(prob) 
                  for prob, idx in zip(top3_prob, top3_idx)}
        
        return results

def create_gradio_interface():
    predictor = CIFAR10Predictor(Config.MODEL_SAVE_PATH)
    
    interface = gr.Interface(
        fn=predictor.predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="CIFAR-10 Image Classifier",
        description="Upload an image to classify it into one of 10 categories: " +
                   ", ".join(Config.CLASSES),
        examples=[
            ["examples/plane.jpg"],
            ["examples/car.jpg"],
            ["examples/cat.jpg"]
        ]
    )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)