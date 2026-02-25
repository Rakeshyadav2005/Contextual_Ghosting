import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageGhoster:
    def __init__(self):
        # Professional Touch: Check if weights are already loaded
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.undo_transform = transforms.ToPILImage()

    def apply_pgd_protection(self, image_data, epsilon=0.01, alpha=0.002, iters=10):
        # image_data can now be a file path OR a PIL Image object from Streamlit
        if isinstance(image_data, str):
            img = Image.open(image_data).convert('RGB')
        else:
            img = image_data.convert('RGB')

        img_t = self.transform(img).unsqueeze(0)
        original_img = img_t.clone().detach() 
        
        adv_img = img_t.clone().detach()
        
        # PGD Core Loop
        for i in range(iters):
            adv_img.requires_grad = True
            outputs = self.model(adv_img)
            
            _, target = torch.max(outputs, 1) 
            loss = nn.CrossEntropyLoss()(outputs, target)
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                adv_img = adv_img + alpha * adv_img.grad.data.sign()
                eta = torch.clamp(adv_img - original_img, min=-epsilon, max=epsilon)
                adv_img = torch.clamp(original_img + eta, min=0, max=1).detach()
            
        return self.undo_transform(adv_img.squeeze(0))