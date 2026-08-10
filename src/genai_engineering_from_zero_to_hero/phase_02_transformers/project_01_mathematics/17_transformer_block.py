import torch
import torch.nn as nn

def describe_tensor(name: str, x: torch.Tensor) -> None:
    print(f"\n{name}")
    print("-" * 60)
    print(f"Shape : {tuple(x.shape)}")
    print(f"Mean  : {x.mean().item():.6f}")
    print(f"Std   : {x.std().item():.6f}")
    print(f"Min   : {x.min().item():.6f}")
    print(f"Max   : {x.max().item():.6f}")
    print(f"NaN   : {torch.isnan(x).any().item()}")
    print(f"Inf   : {torch.isinf(x).any().item()}")


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # (B, T, d_model)
        #       ↓
        # (B, H, T, head_dim)
        q = q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / (self.head_dim ** 0.5)

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # Weighted values
        output = torch.matmul(
            attention_weights,
            v,
        )

        # (B, H, T, head_dim)
        #       ↓
        # (B, T, d_model)
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # ---------------------------------------
        # Multi-Head Attention
        # ---------------------------------------
        describe_tensor("Input", x)
        attention_output = self.attention(x)

        # ---------------------------------------
        # Residual Connection
        # ---------------------------------------
        describe_tensor(
                "Attention Output",
                attention_output,
            )
        x = x + attention_output

        describe_tensor(
                "After First Residual",
                x,
            )
        # ---------------------------------------
        # LayerNorm
        # ---------------------------------------

        x = self.norm1(x)

        describe_tensor(
            "After First LayerNorm",
            x,
        )

        # ---------------------------------------
        # Feed Forward Network
        # ---------------------------------------

        ffn_output = self.ffn(x)

        describe_tensor(
                "FFN Output",
                ffn_output,
            )

        # ---------------------------------------
        # Residual Connection
        # ---------------------------------------

        x = x + ffn_output

        describe_tensor(
                "After Second Residual",
                x,
            )

        # ---------------------------------------
        # LayerNorm
        # ---------------------------------------

        x = self.norm2(x)

        describe_tensor(
            "Final Output",
            x,
        )

        return x


def main():

    # ---------------------------------------
    # Configuration
    # ---------------------------------------

    batch_size = 2
    sequence_length = 4
    d_model = 8
    n_heads = 2
    d_ff = 32

    # ---------------------------------------
    # Input
    # ---------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    print("Input Shape:")
    print(x.shape)

    # ---------------------------------------
    # Create Transformer Block
    # ---------------------------------------

    transformer_block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )

    # ---------------------------------------
    # Forward Pass
    # ---------------------------------------

    output = transformer_block(x)

    print("\nOutput Shape:")
    print(output.shape)

    # ---------------------------------------
    # Verification
    # ---------------------------------------

    print("\nInput and Output Shape Match:")
    print(x.shape == output.shape)

    print("\nContains NaN:")
    print(torch.isnan(output).any())

    print("\nOutput:")
    print(output)


if __name__ == "__main__":
    main()