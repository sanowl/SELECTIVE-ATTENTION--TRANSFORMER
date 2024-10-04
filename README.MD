# Selective Attention Implementation

This repository contains an implementation of a **Selective Attention** mechanism in PyTorch, inspired by the Transformer model's attention mechanism. Selective Attention aims to enhance the efficiency and relevance of the attention process by selectively attending to the most important elements in the input sequence.

## Overview

The implementation includes:
- A `SelectiveAttention` class, which extends `nn.Module`.
- The class accepts input queries (`Q`), keys (`K`), and values (`V`), along with an optional selective mask to control attention behavior.
- Multiple test cases that ensure correctness, handle edge cases, and validate the selective attention functionality.

## Features
- **Multi-head Attention**: The `SelectiveAttention` module splits attention into multiple heads, allowing the model to capture different relationships in the sequence.
- **Selective Masking**: Masks can be used to focus attention only on the relevant tokens, enhancing model efficiency.
- **Dropout Regularization**: Dropout is applied to the attention weights to help prevent overfitting.

## Mathematical Explanation

The Selective Attention mechanism is an adaptation of the scaled dot-product attention used in Transformer models. Here is the mathematical formulation:

### Standard Attention Mechanism

In a typical transformer model, the attention mechanism is computed as follows:

**Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V**

Where:
- **Q**: Query matrix of shape `(batch_size, seq_len, d_model)`
- **K**: Key matrix of shape `(batch_size, seq_len, d_model)`
- **V**: Value matrix of shape `(batch_size, seq_len, d_model)`
- **d_k**: Dimensionality of the key vectors (`d_model / n_heads`)

The similarity between each query and key is computed using the dot product, and the result is scaled by the square root of the key dimension (`sqrt(d_k)`) to avoid large values that might cause gradient instability. The softmax function is then applied to obtain attention weights, which are used to compute a weighted sum of the value vectors (`V`).

### Selective Attention Mechanism

In **Selective Attention**, we introduce a masking mechanism that allows certain elements in the sequence to be ignored during the attention computation. The modified attention mechanism can be represented as:

**SelectiveAttention(Q, K, V) = softmax((QK^T) / sqrt(d_k) - F) V**

Where:
- **F**: A selective masking matrix that helps reduce the influence of irrelevant tokens. The mask can take values such as `-inf` to ensure that specific tokens have zero attention weight after applying the softmax.

The mask `F` is used to adjust the logits before applying the softmax function, effectively controlling the focus of attention. By subtracting `F` from the logits, we ensure that certain elements receive negligible or zero weight, depending on the values in `F`.

The output of the attention mechanism is then computed as the weighted sum of the value vectors (`V`), similar to the standard attention mechanism.

### Multi-head Attention

The attention mechanism is divided into multiple heads, which allows the model to learn different aspects of the relationships between tokens. Each head performs the attention calculation independently, and the outputs are concatenated and projected back to the original dimension.

Mathematically, for each head `i`:

**head_i = SelectiveAttention(QW_i^Q, KW_i^K, VW_i^V)**

Where **W_i^Q**, **W_i^K**, and **W_i^V** are learned projection matrices for the queries, keys, and values, respectively. The final output is obtained by concatenating all the heads and applying an output projection matrix.

## Requirements
- Python 3.7+
- PyTorch 1.8+

To install the required packages, run:
```sh
pip install torch
```

## Usage

The `SelectiveAttention` class can be used as follows:

```python
import torch
from torch import nn
from selective_attention import SelectiveAttention

d_model = 512
n_heads = 8
batch_size = 32
seq_len = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the inputs
Q = torch.randn(batch_size, seq_len, d_model).to(device)
K = torch.randn(batch_size, seq_len, d_model).to(device)
V = torch.randn(batch_size, seq_len, d_model).to(device)

# Create a causal mask to prevent attending to future tokens
selective_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=Q.device) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)

# Initialize and run the SelectiveAttention module
selective_attention = SelectiveAttention(d_model, n_heads).to(device)
output = selective_attention(Q, K, V, selective_mask=selective_mask)
print(output.shape)  # Expected output: (batch_size, seq_len, d_model)
```

## Code Structure
- **SelectiveAttention Class**: Implements multi-head selective attention with optional masking to focus on relevant tokens only.
  - **Methods**:
    - `__init__(d_model, n_heads, dropout=0.1)`: Initializes the model, setting up linear transformations and dropout.
    - `forward(Q, K, V, selective_mask=None)`: Computes the attention output, optionally using a selective mask to control attention focus.

## Tests
The following tests are included to ensure proper functionality:
1. **Basic Sanity Check**: Ensures output dimensions are as expected.
2. **Selective Masking Test**: Verifies that the selective mask correctly reduces attention to irrelevant tokens.
3. **Edge Case Testing**: Tests behavior with zero-length sequences and single-token sequences.
4. **Gradient Check**: Verifies gradients are properly calculated during backpropagation.
5. **Random Mask Testing**: Ensures selective masking behaves as expected for different random masks.

## Example Output
The expected output from the module includes tensor shapes consistent with the input dimensions. The following are some expected outputs:
- `(batch_size, seq_len, d_model)` for the standard forward pass.
- Proper handling of edge cases such as zero-length sequences and single-token inputs.
