import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageGhoster:
    def __init__(self):
        # Load the pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.eval()  # Set to evaluation mode
        
        # Standard ImageNet transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # To convert tensors back to viewable images
        self.undo_transform = transforms.ToPILImage()

    def apply_pgd_protection(self, image_path, epsilon=0.01, alpha=0.002, iters=10):
        # 1. Load image and set up
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0)
        original_img = img_t.clone().detach() 
        
        # 2. Iterative loop (PGD Logic)
        adv_img = img_t.clone().detach()
        
        for i in range(iters):
            adv_img.requires_grad = True
            outputs = self.model(adv_img)
            
            # Use the predicted class as the target to push away from it
            _, target = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()(outputs, target)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update step
            with torch.no_grad():
                # Move image in direction of gradient (increasing loss)
                adv_img = adv_img + alpha * adv_img.grad.data.sign()
                
                # The "Project" Step: Ensure noise stays within epsilon of original
                eta = torch.clamp(adv_img - original_img, min=-epsilon, max=epsilon)
                adv_img = torch.clamp(original_img + eta, min=0, max=1).detach()
            
        return self.undo_transform(adv_img.squeeze(0))