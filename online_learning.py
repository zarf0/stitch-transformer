# online_learning.py
import torch
import torch.nn.functional as F
import json
import os
from datetime import datetime

class OnlineLearning:
    def __init__(self, model, optimizer, save_dir="feedback_data"):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.feedback_buffer = []
        
    def save_correction(self, original_stitches, corrected_stitches, prompt=None, image=None):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "image": image,
            "original_stitches": original_stitches,
            "corrected_stitches": corrected_stitches
        }
        self.feedback_buffer.append(feedback_entry)
        filename = f"{self.save_dir}/feedback_{int(datetime.now().timestamp())}.json"
        with open(filename, 'w') as f:
            json.dump(feedback_entry, f)
        print(f"Saved feedback: {filename}")
    
    def contrastive_loss(self, original, corrected, margin=1.0):
        orig_loss = F.mse_loss(original, corrected, reduction='mean')
        model_orig = self.model(original)
        model_corr = self.model(corrected)
        loss = F.mse_loss(model_corr, corrected) + F.mse_loss(model_orig, original) * 0.1
        return loss
    
    def fine_tune_from_feedback(self, epochs=5):
        if not self.feedback_buffer:
            print("No feedback to learn from.")
            return
        
        print(f"Fine-tuning on {len(self.feedback_buffer)} feedback samples...")
        for epoch in range(epochs):
            total_loss = 0.0
            for feedback in self.feedback_buffer:
                original = torch.tensor(feedback["original_stitches"]).float()
                corrected = torch.tensor(feedback["corrected_stitches"]).float()
                self.optimizer.zero_grad()
                loss = self.contrastive_loss(original, corrected)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.feedback_buffer):.4f}")
        
        self.feedback_buffer = []
        print("Fine-tuning complete!")