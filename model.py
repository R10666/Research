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