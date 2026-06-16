import torch
import torch.nn as nn
from torch.nn import functional as F

# Configuration for GTX 1650 (4GB VRAM)
# We use 8 heads for "Thinking" resolution and 6 layers for depth.
n_embd = 384
n_head = 8
n_layer = 6
block_size = 256
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    """ One head of self-attention: Looking for specific patterns """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ 8 Heads running in parallel (Syntax, Variables, English, Logic, etc.) """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ The reasoning layer where the AI 'thinks' about the gathered data """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ One full 'Thought Block': Communication then Computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BinaryBrainV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(256, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, 256)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def get_fingerprint(self, tokens):
        """ Generates a 'Normalized Vector Identity' for the memory system """
        self.eval()
        with torch.no_grad():
            # Focus on the most recent 128 context window for precision
            context = tokens[-128:] if len(tokens) > 128 else tokens
            # Forward pass (only the last token contains the context summary)
            tok_emb = self.token_embedding_table(context.unsqueeze(0))
            pos_emb = self.position_embedding_table(torch.arange(context.shape[0], device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            
            # Use the global mean of the hidden states as the fingerprint
            fingerprint = torch.mean(x, dim=1).squeeze()
            return F.normalize(fingerprint, p=2, dim=0)

# Initialize
model = BinaryBrainV2().to(device)
