import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# -------------------------------------------------
# Dimensions
# -------------------------------------------------
B = 2
S = 4
d_model = 768
num_heads = 12
d_head = d_model // num_heads   # 64
V = 5000   # vocabulary size

# -------------------------------------------------
# Input
# -------------------------------------------------
X = torch.randn(B, S, d_model)
print("X.shape =", X.shape)   # (B, S, 768)

# -------------------------------------------------
# LayerNorm (Pre-LN)
# -------------------------------------------------
ln1 = nn.LayerNorm(d_model)
X_norm = ln1(X)
print("X_norm.shape =", X_norm.shape)

# -------------------------------------------------
# Linear projections
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
print("V.shape =", Vv.shape)

# -------------------------------------------------
# Split into heads
# -------------------------------------------------
Q = Q.view(B, S, num_heads, d_head).transpose(1, 2)
K = K.view(B, S, num_heads, d_head).transpose(1, 2)
Vv = Vv.view(B, S, num_heads, d_head).transpose(1, 2)

print("After split into heads:")
print("Q.shape =", Q.shape)   # (B, 12, S, 64)
print("K.shape =", K.shape)
print("V.shape =", Vv.shape)

# -------------------------------------------------
# Attention scores
# -------------------------------------------------
scores = Q @ K.transpose(-2, -1)
print("scores.shape =", scores.shape)   # (B, 12, S, S)

scores = scores / math.sqrt(d_head)

# -------------------------------------------------
# Causal mask
# -------------------------------------------------
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
print("mask.shape =", mask.shape)       # (S, S)

scores = scores.masked_fill(mask, float('-inf'))
print("scores after masking.shape =", scores.shape)

# -------------------------------------------------
# Attention weights
# -------------------------------------------------
attn = torch.softmax(scores, dim=-1)
print("attn.shape =", attn.shape)

# -------------------------------------------------
# Weighted sum of V
# -------------------------------------------------
Z = attn @ Vv
print("Z.shape =", Z.shape)   # (B, 12, S, 64)

# -------------------------------------------------
# Recombine heads
# -------------------------------------------------
Z = Z.transpose(1, 2).contiguous().view(B, S, d_model)
print("After recombining heads, Z.shape =", Z.shape)   # (B, S, 768)

# -------------------------------------------------
# Output projection
# -------------------------------------------------
AttnOut = W_O(Z)
print("AttnOut.shape =", AttnOut.shape)   # (B, S, 768)

# -------------------------------------------------
# Residual connection
# -------------------------------------------------
X2 = X + AttnOut
print("X2.shape =", X2.shape)

# -------------------------------------------------
# Second LayerNorm
# -------------------------------------------------
ln2 = nn.LayerNorm(d_model)
X3 = ln2(X2)
print("X3.shape =", X3.shape)

# -------------------------------------------------
# Feedforward network
# -------------------------------------------------
ff1 = nn.Linear(d_model, 3072)
ff2 = nn.Linear(3072, d_model)
gelu = nn.GELU()

H = ff1(X3)
print("After ff1, H.shape =", H.shape)   # (B, S, 3072)

H = gelu(H)
print("After GELU, H.shape =", H.shape)

FFNOut = ff2(H)
print("FFNOut.shape =", FFNOut.shape)    # (B, S, 768)

# -------------------------------------------------
# Second residual connection
# -------------------------------------------------
Output = X2 + FFNOut
print("Output.shape =", Output.shape)    # (B, S, 768)

# -------------------------------------------------
# Final logits layer
# -------------------------------------------------
W_out = nn.Linear(d_model, V, bias=False)
logits = W_out(Output)
print("logits.shape =", logits.shape)    # (B, S, V)

# -------------------------------------------------
# Example target tensor
# Each entry is an integer token ID in [0, V-1]
# Shape: (B, S)
# -------------------------------------------------
targets = torch.tensor([
    [10, 25, 300, 7],
    [42, 100, 5, 999]
], dtype=torch.long)

print("targets.shape =", targets.shape)  # (B, S)

# -------------------------------------------------
# Flatten logits and targets for cross-entropy
# logits:  (B, S, V) -> (B*S, V)
# targets: (B, S)    -> (B*S)
# -------------------------------------------------
logits_flat = logits.view(B * S, V)
targets_flat = targets.view(B * S)

print("logits_flat.shape =", logits_flat.shape)    # (B*S, V)
print("targets_flat.shape =", targets_flat.shape)  # (B*S,)

# -------------------------------------------------
# Cross-entropy loss
# This computes the loss at every sequence position
# and returns the average over all B*S positions
# -------------------------------------------------
loss = F.cross_entropy(logits_flat, targets_flat)
print("loss =", loss.item())

# -------------------------------------------------
# Optional: show the first few target IDs and
# explain which logits row is compared with which target
# -------------------------------------------------
print("\nExample alignment:")
for i in range(min(4, B * S)):
    print(f"logits_flat[{i}].shape = {logits_flat[i].shape}, target = {targets_flat[i].item()}")