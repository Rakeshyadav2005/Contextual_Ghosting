import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageGhoster:
    def __init__(self):
        # Load pre-trained ResNet50 to find image vulnerabilities
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.undo_transform = transforms.ToPILImage()

    def apply_protection(self, image_path, epsilon=0.02):
        # Prepare image for the adversarial engine
        raw_img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(raw_img).unsqueeze(0)
        img_tensor.requires_grad = True 
        
        # Calculate gradients
        output = self.model(img_tensor)
        loss = nn.CrossEntropyLoss()(output, output.max(1)[1])
        
        self.model.zero_grad()
        loss.backward()
        
        # Apply FGSM: Add noise to the pixels
        ghosted_tensor = img_tensor + epsilon * img_tensor.grad.data.sign()
        ghosted_tensor = torch.clamp(ghosted_tensor, 0, 1)
        
        return self.undo_transform(ghosted_tensor.squeeze(0))