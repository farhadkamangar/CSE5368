import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# -------------------------------------------------
# Small dimensions for teaching
# -------------------------------------------------
B = 1
S = 3
d_model = 6
num_heads = 2
d_head = d_model // num_heads   # 3
V = 5                           # vocabulary size

# -------------------------------------------------
# Input tensor
# Shape: (B, S, d_model)
# -------------------------------------------------
X = torch.randn(B, S, d_model)
print("X.shape =", X.shape)
print(X)
print()

# -------------------------------------------------
# First LayerNorm
# -------------------------------------------------
ln1 = nn.LayerNorm(d_model)
X_norm = ln1(X)
print("X_norm.shape =", X_norm.shape)
print(X_norm)
print()

# -------------------------------------------------
# Q, K, V projections
# -------------------------------------------------
W_Q = nn.Linear(d_model, d_model, bias=False)
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)
W_O = nn.Linear(d_model, d_model, bias=False)

Q = W_Q(X_norm)
K = W_K(X_norm)
Vv = W_V(X_norm)

print("Q.shape =", Q.shape)
print("K.shape =", K.shape)
print("Vv.shape =", Vv.shape)
print()

# -------------------------------------------------
# Split into heads
# (B, S, d_model) -> (B, num_heads, S, d_head)
# -------------------------------------------------
Q = Q.view(B, S, num_heads, d_head).transpose(1, 2)
K = K.view(B, S, num_heads, d_head).transpose(1, 2)
Vv = Vv.view(B, S, num_heads, d_head).transpose(1, 2)

print("After split into heads:")
print("Q.shape =", Q.shape)
print("K.shape =", K.shape)
print("Vv.shape =", Vv.shape)
print()

# -------------------------------------------------
# Attention scores
# -------------------------------------------------
scores = Q @ K.transpose(-2, -1)
print("scores.shape =", scores.shape)
print(scores)
print()

scores = scores / math.sqrt(d_head)
print("scaled scores.shape =", scores.shape)
print(scores)
print()

# -------------------------------------------------
# Causal mask
# -------------------------------------------------
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
print("mask.shape =", mask.shape)
print(mask)
print()

scores = scores.masked_fill(mask, float('-inf'))
print("scores after masking:")
print(scores)
print()

# -------------------------------------------------
# Attention weights
# -------------------------------------------------
attn = torch.softmax(scores, dim=-1)
print("attn.shape =", attn.shape)
print(attn)
print()

# -------------------------------------------------
# Weighted sum with V
# -------------------------------------------------
Z = attn @ Vv
print("Z.shape =", Z.shape)
print(Z)
print()

# -------------------------------------------------
# Recombine heads
# (B, num_heads, S, d_head) -> (B, S, d_model)
# -------------------------------------------------
Z = Z.transpose(1, 2).contiguous().view(B, S, d_model)
print("After recombining heads, Z.shape =", Z.shape)
print(Z)
print()

# -------------------------------------------------
# Output projection
# -------------------------------------------------
AttnOut = W_O(Z)
print("AttnOut.shape =", AttnOut.shape)
print(AttnOut)
print()

# -------------------------------------------------
# First residual connection
# -------------------------------------------------
X2 = X + AttnOut
print("X2.shape =", X2.shape)
print(X2)
print()

# -------------------------------------------------
# Second LayerNorm
# -------------------------------------------------
ln2 = nn.LayerNorm(d_model)
X3 = ln2(X2)
print("X3.shape =", X3.shape)
print(X3)
print()

# -------------------------------------------------
# Feedforward network
# -------------------------------------------------
ff1 = nn.Linear(d_model, 8)    # small hidden size for classroom use
ff2 = nn.Linear(8, d_model)
gelu = nn.GELU()

H = ff1(X3)
print("After ff1, H.shape =", H.shape)
print(H)
print()

H = gelu(H)
print("After GELU, H.shape =", H.shape)
print(H)
print()

FFNOut = ff2(H)
print("FFNOut.shape =", FFNOut.shape)
print(FFNOut)
print()

# -------------------------------------------------
# Second residual connection
# -------------------------------------------------
Output = X2 + FFNOut
print("Output.shape =", Output.shape)
print(Output)
print()

# -------------------------------------------------
# Final logits layer
# (B, S, d_model) -> (B, S, V)
# -------------------------------------------------
W_out = nn.Linear(d_model, V, bias=False)
logits = W_out(Output)
print("logits.shape =", logits.shape)
print(logits)
print()

# -------------------------------------------------
# Example targets
# Shape: (B, S)
# These are token IDs in [0, V-1]
# -------------------------------------------------
targets = torch.tensor([[2, 1, 3]], dtype=torch.long)
print("targets.shape =", targets.shape)
print(targets)
print()

# -------------------------------------------------
# Flatten for cross-entropy
# logits:  (B, S, V) -> (B*S, V)
# targets: (B, S)    -> (B*S)
# -------------------------------------------------
logits_flat = logits.view(B * S, V)
targets_flat = targets.view(B * S)

print("logits_flat.shape =", logits_flat.shape)
print(logits_flat)
print()

print("targets_flat.shape =", targets_flat.shape)
print(targets_flat)
print()

# -------------------------------------------------
# Cross-entropy loss
# -------------------------------------------------
loss = F.cross_entropy(logits_flat, targets_flat)
print("loss =", loss.item())
print()

# -------------------------------------------------
# Show alignment between logits rows and targets
# -------------------------------------------------
print("Alignment between predictions and targets:")
for i in range(B * S):
    print(f"Position {i}: logits_flat[{i}] predicts target {targets_flat[i].item()}")