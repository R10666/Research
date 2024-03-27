#https://www.youtube.com/watch?v=ISNdQcPhsts&list=PLGgEpZvMWvD_2OBT9UwmC7q0uxzk_4wGT&index=8&t=3562s

##transformer model##

#input embedding
#vector of size 512

import math
import torch #tensor calculation
import torch.nn as nn

##Token embedding##
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, volcab_size: int): #constructor 
        super().__int__()
        self.d_model = d_model
        self.volcab_size = volcab_size
        self.embedding = nn.Embedding(volcab_size, d_model) # word inputs -> vectors


    def forward(self, x)
        return self.embedding(x) * math.sqrt(embedding)


##positional encoding##
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #dropout - reduce dependency on neaurons to help model learn


        # Create  a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype = torch)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #sin for even, cos for odd
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe) # save the positional encoding as a buffer

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #add positional encoding to itself, also no learn since fixed
        return self.dropout(x)



class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None: #eps to make math stable if s.d is very small
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiply ,nn.Parameter allows for learning
        self.bias = nn.Parameter(torch.zeros(1)) #added

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std = x.std(dim = -1, keepdim = True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias #formula for normalization 

    class FeedFordwardBlock(nn.Module):

        def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
            super().__init__()
            self.linear_1 = nn.Linear(d.model, d_ff) #W1 and B1
            self.dropout = nn.Dropout(dropout)
            self.linear_2 nn.Linear(d_ff, d_model) # W2 and B2

        def forward(self, x):
            # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h #h is number of heads
        assert d_model % h == 0, "d_model is not divisiable by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model) --> (Batch, h, Seq_Len, d_k)
        query = quary.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)











