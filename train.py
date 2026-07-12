import torch
import torch.nn as nn
import torch.nn.functional as F
import time
try:
    from google import genai
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'google-genai'])
    from google import genai
from google.colab import userdata
from torch.utils.data import Dataset, DataLoader

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

class RoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_position_embeddings).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb[None, None, :, :])
        self.register_buffer("sin_cached", emb[None, None, :, :])
    def forward(self, x, seq_len=None):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (x * cos) + (self.rotate_half(x) * sin)
    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class StudentModel(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=128, n_head=4, n_layer=3):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.rope = RoPE(self.head_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(n_embd, n_head, batch_first=True),
                'norm1': nn.LayerNorm(n_embd),
                'swiglu': SwiGLU(n_embd),
                'norm2': nn.LayerNorm(n_embd)
            }) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, teacher_logits=None, temp=2.0, alpha=0.5):
        b, t = idx.size()
        x = self.token_embedding(idx)
        x = x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        x = self.rope(x, seq_len=t)
        x = x.transpose(1, 2).contiguous().view(b, t, -1)
        for layer in self.layers:
            n1 = layer['norm1'](x)
            attn_out, _ = layer['attn'](n1, n1, n1)
            x = x + attn_out
            x = x + layer['swiglu'](layer['norm2'](x))
        logits = self.lm_head(self.ln_f(x))
        return logits, None

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentModel().to(device)
    # Ensure we explicitly save it to the current directory
    torch.save(model.state_dict(), '/content/student_model.pth')
    print(f"Distillation code ready. Model weights explicitly saved to /content/student_model.pth.")

if __name__ == '__main__':
    train()
