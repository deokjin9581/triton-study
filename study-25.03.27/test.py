import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.optim as optim

# -----------------------------------------------------------------------------
# Triton Kernel (forward only)
# -----------------------------------------------------------------------------
@triton.jit
def linear_kernel(X_ptr, W_ptr, B_ptr, Y_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, P: tl.constexpr):
    """
    Computes Y = X @ W + B
    X: [M, K] (row-major)
    W: [K, P] (padded weight, where first N columns are valid)
    B: [P] (padded bias, where first N entries are valid)
    Y: [M, N] (output)
    M, N, K: dimensions (with N possibly not a power-of-2)
    BLOCK_M: block size along M (must be a power-of-2)
    P: padded number of columns (next power-of-two >= N)
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    # Compute row indices for the block.
    row_ids = row_start + tl.arange(0, BLOCK_M)  # shape: (BLOCK_M,)

    # Create a padded range for columns.
    offs_n = tl.arange(0, P)

    # Allocate accumulator with padded column dimension.
    acc = tl.zeros((BLOCK_M, P), dtype=tl.float32)

    # Loop over input dimension.
    for k in range(K):
        x_val = tl.load(X_ptr + row_ids * K + k)  # shape: (BLOCK_M,)
        w_val = tl.load(W_ptr + k * P + offs_n, mask=offs_n < N)
        acc += x_val[:, None] * w_val[None, :]

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N)
    acc += bias[None, :]

    mask_row = row_ids < M
    mask_col = offs_n < N
    tl.store(Y_ptr + row_ids[:, None] * N + offs_n,
             acc,
             mask=mask_row[:, None] & mask_col[None, :])

# -----------------------------------------------------------------------------
# Custom Autograd Function wrapping our Triton Kernel
# -----------------------------------------------------------------------------
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, block_size):
        # X: [M, K]
        # weight: [K, N]
        # bias: [N]
        M, K = X.shape
        N = weight.shape[1]

        # Compute padded dimension P (next power-of-two).
        def next_power_of_2(n):
            return 1 << (n - 1).bit_length()
        P = next_power_of_2(N)

        # Pad weight and bias if necessary.
        if P != N:
            W_padded = torch.zeros(K, P, device=X.device, dtype=torch.float32)
            W_padded[:, :N] = weight
            bias_padded = torch.zeros(P, device=X.device, dtype=torch.float32)
            bias_padded[:N] = bias
        else:
            W_padded = weight
            bias_padded = bias

        Y = torch.empty((M, N), device=X.device, dtype=torch.float32)
        grid = (triton.cdiv(M, block_size),)

        linear_kernel[grid](
            X.contiguous(),
            W_padded.contiguous(),
            bias_padded.contiguous(),
            Y,
            int(M), int(N), int(K),
            BLOCK_M=block_size,
            P=P
        )
        # Save for backward.
        ctx.save_for_backward(X, weight, bias)
        ctx.block_size = block_size
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors.
        X, weight, bias = ctx.saved_tensors

        grad_X = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_X = grad_output.matmul(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = X.t().matmul(grad_output)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        # The block_size does not require grad.
        return grad_X, grad_weight, grad_bias, None

# -----------------------------------------------------------------------------
# Custom Module Using the Autograd Function
# -----------------------------------------------------------------------------
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=16):
        """
        in_features: size of input features (K)
        out_features: size of output features (N)
        block_size: number of rows processed per kernel instance (BLOCK_M),
                    recommended to be a power-of-two (e.g. 16 or 32)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        return TritonLinearFunction.apply(X, self.weight, self.bias, self.block_size)

# -----------------------------------------------------------------------------
# Example Model Using TritonLinear for Both Layers
# -----------------------------------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self, num_embeddings=10, embedding_dim=4, num_classes=10):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = TritonLinear(embedding_dim, 16)   # 16 is power-of-two (no padding needed)
        self.fc2 = TritonLinear(16, num_classes)      # num_classes may not be power-of-two
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)            # shape: [batch_size, embedding_dim]
        x = self.activation(self.fc1(x)) # shape: [batch_size, 16]
        x = self.fc2(x)                  # shape: [batch_size, num_classes]
        return x

# -----------------------------------------------------------------------------
# Testing the Implementation
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel(num_embeddings=20, embedding_dim=4, num_classes=10).to(device)
    x = torch.randint(0, 20, (8,), dtype=torch.long).to(device)  # example batch of indices

    # Forward pass.
    output = model(x)
    print("Model output:", output)

    # Test backward pass.
    output.sum().backward()
    print("Backward pass completed successfully.")
    
    
    model = SimpleModel().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 더미 데이터 생성
    num_samples = 100
    torch.manual_seed(42)
    x = torch.randint(0, 10, (num_samples,))  # 정수 인덱스 데이터
    y = (x * 2 + 3) % 10  # 단순 선형 관계 (y = 2x + 3) 후 클래스 변환

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 GPU로 이동
    x, y = x.to(device), y.to(device)

    # 학습 루프
    num_epochs = 1000
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
