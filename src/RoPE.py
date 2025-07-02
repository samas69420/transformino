import torch
from config import ROPE_MAX_LEN

class RoPE(torch.nn.Module):

    def __init__(self, head_dim, base_freq = 10000, max_len = ROPE_MAX_LEN):

        super().__init__()

        positions = torch.arange(max_len, dtype=torch.float)
        thetas = base_freq ** (-torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)

        # do all possible combinations between thetas and positions using outer product
        freqs = positions.unsqueeze(1) @ thetas.unsqueeze(0)

        # precompute sin and cos for all the frequencies, including doubling
        self.cos = freqs.cos().repeat_interleave(2, dim=-1)
        self.sin = freqs.sin().repeat_interleave(2, dim=-1)
    
    def forward(self, X):

        def rotate_half(X):

            # turn each row into a matrix of len 2 and stack all matrices obtained
            X_pairs = X.view(*X.shape[:-1], -1, 2)  

            # swap the pairs and change sign
            X_rotated = torch.stack([-X_pairs[..., 1], X_pairs[..., 0]], dim=-1)

            # Reshape back to the original form
            # now each row of the matrices is a "row_hat"
            return X_rotated.view(*X.shape)

        # rotate every piece of row of the matrix X according to the RoPE algorithm
        return (X * self.cos[:X.shape[-2],:X.shape[-1]].to(X.device)) + (rotate_half(X) * self.sin[:X.shape[-2],:X.shape[-1]].to(X.device))
