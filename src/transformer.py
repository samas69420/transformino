import torch
from RoPE import RoPE
from config import MODEL_MLP_HIDDEN_LAYERS,\
                   MODEL_MLP_HIDDEN_LAYER_SIZE,\
                   MODEL_SIZE, DESIRED_VOCAB_LEN


class MultiHeadAttention(torch.nn.Module):
    """
    efficient version of MHA that doesn't instantiate every single head and loop 
    over all of them but process them in parallel, always expect batched input
    """

    def __init__(self, model_size, n_heads, head_dim):

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.model_size = model_size

        # single large Linear layer for all Q, K, V projections for all heads
        self.W_qkv = torch.nn.Linear(model_size, 3 * n_heads * head_dim, bias=False)
        
        self.W_o = torch.nn.Linear(n_heads * head_dim, model_size, bias=False)
        
        self.rope = RoPE(self.head_dim)

    def forward(self, x):

        # x shape: [batch_size, seq_len, model_size]
        batch_size, seq_len, _ = x.shape

        # qkv shape: [batch_size, seq_len, 3 * n_heads * head_dim]
        qkv = self.W_qkv(x)

        # without considering batch we have that:
        #              _                                             _ 
        #             |               |               |               |
        # X * W_qkv = | Q1...Qn_heads | K1...Kn_heads | V1...Vn_heads |
        #             |_              |               |              _|
        #                                              _          _
        #                                        ^    |            |
        # the dim of Q/K/V for a single head  seq len |< head dim >|
        #                                        v    |_          _|

        # split into Q, K, V
        # the split dimension is the last one, output shape for all q,k,v 
        # tensors will be [batch, seq_len, n_heads * head_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to -> [batch, seq_len, n_heads, head_dim] (separate heads)
        # then transpose to -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # apply positional embedding to queries and keys
        q = self.rope(q)
        k = self.rope(k)

        # assume seq_len is the same for q, k, v (its self attention)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0).to(x.device)
        
        # output shape: [batch, n_heads, seq_len, head_dim]
        # basically the same as softmax((q @ k.mT)/sqrt(head_dim)) @ v
        # but more efficient 
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

        # Transpose back to -> [batch, seq_len, n_heads, head_dim]
        # Then reshape to -> [batch, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(output)
        

class MLP(torch.nn.Module):

    def __init__(self, model_size):

        super().__init__()
        
        inp_size = model_size
        self.layers = []
        
        for _ in range(MODEL_MLP_HIDDEN_LAYERS):

            self.layers.append(torch.nn.Linear(in_features = inp_size,
                                out_features = MODEL_MLP_HIDDEN_LAYER_SIZE))

            self.layers.append(torch.nn.GELU())

            inp_size = MODEL_MLP_HIDDEN_LAYER_SIZE

        self.layers.append(torch.nn.Linear(in_features = MODEL_MLP_HIDDEN_LAYER_SIZE,
                                           out_features = model_size))

        self.layers = torch.nn.ModuleList(self.layers)


    def forward(self, x):
        
        y = x

        for layer in self.layers:
            y = layer(y)

        return y
        
        
class BaseBlock(torch.nn.Module):

    def __init__(self, model_size, n_heads, head_dim):

        super().__init__()

        self.mha = MultiHeadAttention(model_size, n_heads, head_dim)

        self.mlp = MLP(model_size)

        self.layernorm1 = torch.nn.LayerNorm(MODEL_SIZE)
        self.layernorm2 = torch.nn.LayerNorm(MODEL_SIZE)


    def forward(self, x):
        
        x = x
        
        x = self.mha(self.layernorm1(x)) + x

        x = self.mlp(self.layernorm2(x)) + x

        return x


class Transformer(torch.nn.Module):
    
    def __init__(self,model_size, n_blocks, n_heads, head_dim):
        
        super().__init__()

        self.layers = []

        self.layers.append(torch.nn.Embedding(DESIRED_VOCAB_LEN, model_size))

        for _ in range(n_blocks):
            self.layers.append(BaseBlock(model_size, n_heads, head_dim))    

        self.layers.append(torch.nn.Linear(in_features = model_size, out_features = DESIRED_VOCAB_LEN))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self,x):

        for layer in self.layers:
            x = layer(x)

        return x

