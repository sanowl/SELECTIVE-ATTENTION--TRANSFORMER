import torch
import torch.nn.functional as F
from torch import nn

class SelectiveAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelectiveAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, selective_mask=None):
        batch_size, seq_len, _ = Q.size()

        Q = self.query(Q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(K).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(V).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=Q.device))

        if selective_mask is not None:
            attn_logits = attn_logits.masked_fill(selective_mask == float('-inf'), float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attention_output = torch.matmul(attn_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out(attention_output)

        return output

d_model = 512
n_heads = 8
batch_size = 32
seq_len = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = torch.randn(batch_size, seq_len, d_model).to(device)
K = torch.randn(batch_size, seq_len, d_model).to(device)
V = torch.randn(batch_size, seq_len, d_model).to(device)

selective_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=Q.device) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)

selective_attention = SelectiveAttention(d_model, n_heads).to(device)
output = selective_attention(Q, K, V, selective_mask=selective_mask)
print(output.shape)

assert output.shape == (batch_size, seq_len, d_model), f"Unexpected output shape: {output.shape}"
assert not torch.isnan(output).any(), "Output contains NaN values"
assert not torch.isinf(output).any(), "Output contains Inf values"

mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=Q.device).unsqueeze(0).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)
masked_output = selective_attention(Q, K, V, selective_mask=mask)
print(masked_output.shape)

try:
    zero_seq_len = 0
    Q_zero = torch.randn(batch_size, zero_seq_len, d_model).to(device)
    K_zero = torch.randn(batch_size, zero_seq_len, d_model).to(device)
    V_zero = torch.randn(batch_size, zero_seq_len, d_model).to(device)
    output_zero = selective_attention(Q_zero, K_zero, V_zero)
    print(f"Zero-length sequence output shape: {output_zero.shape}")
except Exception as e:
    print(f"Zero-length sequence test failed: {e}")

single_seq_len = 1
Q_single = torch.randn(batch_size, single_seq_len, d_model).to(device)
K_single = torch.randn(batch_size, single_seq_len, d_model).to(device)
V_single = torch.randn(batch_size, single_seq_len, d_model).to(device)
output_single = selective_attention(Q_single, K_single, V_single)
print(f"Single-token sequence output shape: {output_single.shape}")

output = selective_attention(Q, K, V, selective_mask=selective_mask)
loss = output.sum()
loss.backward()
for param in selective_attention.parameters():
    assert param.grad is not None, "Gradient is not computed for parameter"

random_mask = torch.randint(0, 2, (seq_len, seq_len), dtype=torch.bool, device=Q.device).unsqueeze(0).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)
output_with_random_mask = selective_attention(Q, K, V, selective_mask=random_mask)
print(output_with_random_mask.shape)