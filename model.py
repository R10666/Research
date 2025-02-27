#https://www.youtube.com/watch?v=ISNdQcPhsts&list=PLGgEpZvMWvD_2OBT9UwmC7q0uxzk_4wGT&index=8&t=3562s

##transformer model##

#input embedding
#vector of size 512


import math
import torch #tensor calculation
import torch.nn as nn



##Token embedding##
##Takes the Token IDs and apply the input embeddings (now you have a matrix)##
#convert input into vector of size 512
class InputEmbeddings(nn.Module):
    ## this function doesn't retun anything ##
    ## first ststameent here as 'self' ignored while in use -- (d_model,vocab_size) ##
    def __init__(self, d_model: int, vocab_size: int) -> None: #constructor 
        super().__init__() #inheritance
        self.d_model = d_model  # d_model is the size/dimention of the vector
        self.vocab_size = vocab_size #number of words in the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model) # word inputs (Token ID) -> vectors
        ## how does nn.Embedding access the semantic information? Is it language specific? ##

    ## this function takes input and ouputs for this class ##    
    def forward(self, x):  # forward is the process that's run after constructor initializes the parameters
        return self.embedding(x) * math.sqrt(self.d_model) #multiply weight of embedding by sqrt(d_model) this is how is done on the papers

########################


##positional encoding##
# Add a vector of same size onto the embedding to repersent the position
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: # -> None means return nothing, like void() *it's Not required in python, only for understanding purpose
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len #max length of the sentence
        self.dropout = nn.Dropout(dropout) #dropout is a hyperparameter - reduce dependency on neurons to help model learn
        # drop out prevents overfitting and makes the model more generalized, this is done by setting a random set of neurons to zero
        # probability density of dropping a neuron, defined to be between 0 and 1 


        # Create a matrix of shape (seq_len, d_model)
        #seq_len number of column, where each column is a d_model(each word has its own d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1), it's a vertical vector
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # ^this is simplified from origional to make equation more numerically stable, # div_term is acquired as a commonly used division term for mathematical stability 

        #apply sin for even, cos for odd, [start_range:stop_range, start:stop:step]
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model), this is now a tensor
        # ^each matrix is for one sentence, we do this to allow for multiple sentence
        
        self.register_buffer('pe', pe) # save the positional encoding as a buffer
        # ^not saved by default since it's not a train-able parameter



    def forward(self, x):
        #add positional encoding to embedding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # False -> means no learn since this process is fixed 
        return self.dropout(x)              # ^ no learn since fixed
         # it drops out, but WHY?


########################

##Layer Normalization##
# Calculate mean and var of each layer, then calculate mean and var using mean and var of each layer, also introduce bias 
class LayerNormalization(nn.Module):

    def __init__(self, eps:float = 10**-6) -> None: 
        super().__init__()
        self.eps = eps  #eps to make math stable if s.d is very small (eps repersent 0)
        self.alpha = nn.Parameter(torch.ones(1)) #multiply ,nn.Parameter allows for learning
        self.bias = nn.Parameter(torch.zeros(1)) #added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) #calculate the last/latest mean
        std = x.std(dim = -1, keepdim = True)   # "keepdim = True" means that the number used to calculate is kept
        return self.alpha * (x - mean) / (std + self.eps) + self.bias #formula for normalization 

########################

## FF linear block ##
# *Applies two linear transformations and ReLU in between
# ReLU = max(x, 0)  , gives largest value from 0 or x(input)
class FeedForwardBlock(nn.Module):
    ## 2 linear layers as convolutions##
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
    ## 2 linear layers and ReLU activation for just first layer and dropout in between layers but not after second layer##
    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) #Apply ReLU and linear transform

########################


##Multi-Head Attention##
#create Query, key and value from input embedding
#use Query, key and value to calculate and normalize attention score, (done with softmax)
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model #width of matrix 
        self.h = h  #h is number of heads
        assert d_model % h == 0, "d_model is not divisible by h" # make sure the width of matrix can be split evenly

        self.d_k = d_model // h  # each head
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv
        self.w_o = nn.Linear(d_model, d_model) #Wo
        #  ^ These are the weights to perform linear transformation to query, key and value
        self.dropout = nn.Dropout(dropout)

    @staticmethod  # function belong to the class itself instead of a instance, does not need a instance to call
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # brings to the last but WHY and how it applies for a number? 

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # normalization, @ is matrix multiplication
        if mask is not None:   # if masking is defined, not relavent for encoder, but this will work as masked multi head attention for decoder
            attention_scores.masked_fill_(mask == 0, -1e9) # we use a very samll value for mask so after softmax it will be zero
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len), apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores #multiply attention score by value -- end of softmax operation


    ##  SYNTAX NEEDS TO BE UNDERSTOOD CLEARLLY ! (such as matrix muliply and splitting how handled in view funtion) ##
    def forward(self, q, k, v, mask):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # ^ spliting linearly transformed matrix into heads, 
        # head is split along d_model, so embedding is split, each head still have the whole sentence


        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) #dot product, normalize, softmax

        #(Batch, h, seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # putting each head back into a single matrix

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)

########################


##Residual Connection##
# combine two previous layers 
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer): # sublayer is the one before norm layer
        return x + self.dropout(sublayer(self.norm(x))) # note there are implimentation where norm is applied before sublayer

########################

##Encoder block##
# an encoder block of transformer model, this will combine all the other classes together
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__() 
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # ModuleList organize lists of modules

    def forward(self, x, src_mask): #mask prevent padding words to interact with other words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # 1st residual connection, it's the self attention norm
        x = self.residual_connections[1](x, self.feed_forward_block) # 2nd residual connection, the feed forward part merged with the output we just calculated ^
        return x

########################


## WHY THIS PART IS NEEDED AND HOW DOES IT 
##Encoder##
# made of n number of encoder blocks
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:  #apply one layer after another
            x = layer(x, mask)
        return self.norm(x) # the output of this layer become the input for the next

########################

##Decoder Block##
#with some slight changes all the modules from encoder can be reused
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):  #src mask applied for encoder, tgt mask for decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # 1st residual connection at decoder is the self attention 
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) #2nd residual connection is with output of encoder
        x = self.residual_connections[2](x, self.feed_forward_block) # 3rd is with the last feedforward block 
        return x

########################

##Decoder##
# runs n time of the Decoder block
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) # output of one block becomes the input of the next

########################

##Linear layer##
class ProjectionLayer(nn.Module): # projects the embedding back into vocabulary 
    def __init__(self, d_model, vocab_size) -> None:
    #def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_Size)
        return torch.log_softmax(self.proj(x), dim = -1) # apply softmax


########################


##Transformer##
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed #input lang embedding 
        self.tgt_embed = tgt_embed #output lang embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

#Forward()# #we split forward into three for better visualizing, also during inferencing we can reuse the output of the encoder

    def encode(self, src, src_mask):
        src = self.src_embed(src) #embeding
        src = self.src_pos(src) #positional encoding
        return self.encoder(src, src_mask) #encode

    #def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) #decode

    def project(self, x):
        return self.projection_layer(x) #linear transform

###

########################

##Transformer Block Constructor##
#combines everything together
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer: #all numbers from paper
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N): # N number of encoder block, 
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block) # save in array 

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) #has a cross attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) # save in array 

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks)) # create and save 1 instance
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) #other method aviliable too, this is most common

    return transformer 




        









