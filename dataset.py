#This file is for building the dataset
## imports ##
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

##################

## Dataset ##
class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None: #constructor of Dataset
        super().__init__()

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype = torch.int64) # build token into tensor
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype = torch.int64) # token_to_id gets number and then put into tensor
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64)

    # this just define the length of dataset
    def __len__(self):
        return len(self.ds)


    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index] # get the pair of sentence from dataset(sentence in both langauge)
        src_text = src_target_pair["translation"][self.src_lang] #get the sorce lang part
        tgt_text = src_target_pair["translation"][self.tgt_lang] #get the tgt lang part

        # now we split the sentence into words
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # returns a array of input id of each word in the sentence
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # same for tgt lang / decoder

        # since sequence length is fixed, if sentence is too short we fill it with padding([PAD]) tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # minus 2 since there is SOS adn EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # minus 1 since there is only SOS (in training)

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long") # check if sentence is longer than sequence length that we choose

        # Add SOS and EOS to theh source text
        encoder_input = torch.cat( # This builds the input tensor for encoder
            [
                self.sos_token, # add start of sentence token 
                torch.tensor(enc_input_tokens, dtype = torch.int64), # add all token of source text
                self.eos_token,  # add end of sentence
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64) # fills the rest of empty space with padding token
            ]
        )

        decoder_input = torch.cat( # This builds the input tensor for decoder, no end of sentence for this one
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        label = torch.cat( # label is the expected output of decoder, this is sometimes also called target
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # sanity check
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # This is basically what our dataset consistes of:
        return {
            "encoder_input": encoder_input, # (Seq_Len)
            "decoder_input": decoder_input, # (Seq_Len)
            # encoder mask is for padding and decoder mask is for both padding as well as non-causal terms
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len), this is needed since there may be padding in encoder
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            "label": label, # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text    
        }

def causal_mask(size): # this will diagonalize the matrix to mask the future terms
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0