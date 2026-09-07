# vision_encoder.py
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image


class VisionEncoder:
    def __init__(self, model_name="resnet50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            # Remove classification head
            self.model.fc = nn.Identity()
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def encode(self, image_bgr):
        """
        Convert BGR image to embedding.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Resize and normalize
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_tensor)

        return embedding
