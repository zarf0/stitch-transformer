# stitch_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearAttention(nn.Module):
    """Performer-style linear attention for O(n) complexity."""

    def __init__(self, dim, heads=8, dim_head=64, kernel_fn=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.kernel_fn = kernel_fn or F.relu

    def forward(self, x, mask=None):
        B, N, C = x.shape
        H = self.heads
        D = self.dim_head

        qkv = self.to_qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        q = q * self.scale
        k_norm = k / (k.sum(dim=-1, keepdim=True) + 1e-6)

        kv = k_norm.transpose(-2, -1) @ v
        out = q @ kv

        out = out.permute(0, 2, 1, 3).reshape(B, N, H * D)
        return self.to_out(out)


class StitchTransformer(nn.Module):
    def __init__(
        self,
        vocab_dim=5,
        d_model=256,
        nhead=8,
        num_layers=6,
        max_seq_len=20000,
        text_encoder_dim=768,
        image_encoder_dim=2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Text encoder projection
        self.text_proj = nn.Linear(text_encoder_dim, d_model)

        # Image encoder projection
        self.image_proj = nn.Linear(image_encoder_dim, d_model)

        # Fusion layer (combine text + image embeddings)
        self.fusion = nn.Linear(d_model * 2, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Input projection
        self.input_proj = nn.Linear(vocab_dim, d_model)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, text_embedding=None, image_embedding=None):
        """
        x: (batch, seq_len, vocab_dim) - previous stitch tokens
        text_embedding: (batch, text_encoder_dim) - optional text prompt
        image_embedding: (batch, image_encoder_dim) - optional image
        """
        B, S, V = x.shape

        # Project input tokens
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :S, :]

        # Build context embedding (text + image)
        context = None
        if text_embedding is not None and image_embedding is not None:
            text = self.text_proj(text_embedding)
            image = self.image_proj(image_embedding)
            combined = torch.cat([text, image], dim=-1)
            context = self.fusion(combined)
        elif text_embedding is not None:
            context = self.text_proj(text_embedding)
        elif image_embedding is not None:
            context = self.image_proj(image_embedding)
        else:
            raise ValueError("Must provide either text_embedding or image_embedding")

        # FIXED: Ensure context is 3-D (batch, 1, d_model) for TransformerDecoderLayer
        context = context.unsqueeze(1)  # (batch, 1, d_model) - 3D

        # Use context as memory for decoder
        memory = context  # (batch, 1, d_model) - 3D

        # Pass through transformer layers (using memory, NOT context directly)
        for layer in self.layers:
            x = layer(x, memory)

        # Project to next token prediction
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def generate(
        self, text_embedding=None, image_embedding=None, max_len=10000, temperature=1.0
    ):
        """
        Autoregressive generation from text and/or image.
        """
        self.eval()
        with torch.no_grad():
            B = (
                text_embedding.shape[0]
                if text_embedding is not None
                else image_embedding.shape[0]
            )

            # Start with start token
            current = torch.zeros(B, 1, 5).to(self.pos_encoder.device)
            current[:, 0, :] = torch.tensor([0, 0, 0, 0, 0])

            generated_tokens = []

            for _ in range(max_len):
                logits = self.forward(current, text_embedding, image_embedding)
                next_logits = logits[:, -1, :]

                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                if next_token[0, 0].item() == 1 and len(generated_tokens) > 10:
                    break

                next_token_5d = torch.zeros(B, 1, 5).to(self.pos_encoder.device)
                next_token_5d[:, 0, :] = next_token.float()
                current = torch.cat([current, next_token_5d], dim=1)
                generated_tokens.append(next_token_5d[:, 0, :].cpu().numpy())

            return torch.tensor(generated_tokens) if generated_tokens else None
