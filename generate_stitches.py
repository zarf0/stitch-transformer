# generate_stitches.py
import torch
import cv2
import numpy as np

from stitch_transformer import StitchTransformer
from text_encoder import TextEncoder
from vision_encoder import VisionEncoder


class StitchGenerator:
    def __init__(self, model_path="models/stitch_transformer_final.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StitchTransformer()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()

        print("Stitch Generator ready!")

    def from_text(self, prompt, max_len=10000, temperature=1.0):
        """Generate stitches from text prompt."""
        text_embedding = self.text_encoder.encode(prompt)
        with torch.no_grad():
            generated = self.model.generate(
                text_embedding=text_embedding, max_len=max_len, temperature=temperature
            )
        return generated

    def from_image(self, image_path, max_len=10000, temperature=1.0):
        """Generate stitches from image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_embedding = self.vision_encoder.encode(image)
        with torch.no_grad():
            generated = self.model.generate(
                image_embedding=image_embedding,
                max_len=max_len,
                temperature=temperature,
            )
        return generated

    def from_both(self, prompt, image_path, max_len=10000, temperature=1.0):
        """Generate stitches from both text and image."""
        text_embedding = self.text_encoder.encode(prompt)
        image = cv2.imread(image_path)
        if image is not None:
            image_embedding = self.vision_encoder.encode(image)
        else:
            image_embedding = None

        with torch.no_grad():
            generated = self.model.generate(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                max_len=max_len,
                temperature=temperature,
            )
        return generated

    def tokens_to_stitch_blocks(self, tokens):
        """Convert generated tokens to stitch blocks."""
        stitch_blocks = []
        current_block = []

        for token in tokens:
            dx, dy, is_stitch, is_trim, is_color_change = token
            if is_color_change:
                if current_block:
                    stitch_blocks.append(
                        {"stitches": current_block, "color_bgr": (0, 0, 0)}
                    )
                    current_block = []
            elif is_stitch:
                current_block.append((dx, dy))
            elif is_trim:
                if current_block:
                    stitch_blocks.append(
                        {"stitches": current_block, "color_bgr": (0, 0, 0)}
                    )
                    current_block = []

        if current_block:
            stitch_blocks.append({"stitches": current_block, "color_bgr": (0, 0, 0)})

        return stitch_blocks

    def export_to_file(self, stitch_blocks, output_path, formats=("dst", "pes")):
        """Export stitch blocks to embroidery files."""
        from export_embroidery import export_all
        import os

        base_path = os.path.splitext(output_path)[0]
        exported = export_all(stitch_blocks, base_path, formats)
        return exported
