### IMPORTS ###
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split  # allow for loading data and splitting data

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config


from torch.utils.tensorboard import SummaryWriter #Visualisation when training model

import warnings
from tqdm import tqdm

from pathlib import Path # used to define path to files  

from datasets import load_dataset # to get the langauge dataset
from tokenizers import Tokenizer # from hugging face 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

######################

def get_all_sentences(ds, lang):
    for item in ds:  # for the dataset we are using here each item is a pair of sentence, one in sorce lang, and one in tgt lang
        yield item["translation"][lang] #get the wanted langauge from dataset

######################

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang)) # Path to where tokenizer file is saved, format allow for variable inside filename
    if not Path.exists(tokenizer_path):  # create one if file does not exist, file will be generated in current dir
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]")) # if word is not know to tokenizer's vocabulary, it will be given a [UNK] token
        tokenizer.pre_tokenizers = Whitespace() #split by whitespace, so each word is a token
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2) #WordLevelTrainer means one token each word
        #[PAD] = padding, used for training
        #[SOS] = start of sentence
        #[EOS] = end of sentence 
        #min_frequency is the minmium number of time a word has to appear for it to be in the vocabulary
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer) #training
        tokenizer.save(str(tokenizer_ptah)) #save once done
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

######################

def get_ds(config):
    ds_raw = load_dataset("opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train") #get the dataset(training), split means only that part of the dataset is loaded

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"]) #tokenize source
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"]) #tokenize target

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) #this splits the dataset into the two specified size

    # Build the two datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    # Calculates the maximum sentence length of source and target from the tokens:
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # here we take data from our dataset to train and validate, done with DataLoader which iterates over the our datasets in batches
    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True) # shuffle means data is shuffled before batch created 
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True) # batch_size = 1 means that one sentence per batch

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

######################

# simple function to help call and build the model:
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], config["d_model"])
    return model


def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if GPU parallel processing is aviliable
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents = True, exist_ok = True) #find or create the model folder 

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config) #get data
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device) # get model and transfer to device

    #Tensorboard, This bascially gives a visual progress bar and stuff when training
    writer = SummaryWriter(config["experiment_name"])

    # optimizer used to adjust parameters of model to minimize the loss function during training process
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9) # uses the Adam algorthum for optimizing

    initial_epoch = 0 
    global_step = 0 
    if config["preload"]: # This allows for continuing training after a crash or stop, we just need to define a starting epoch
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = statee["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    ## cross entropy loss, measures difference between predicted probability distribution and actual distribution of target labels.
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing = 0.1).to(device) 
    # ^ignore padding when calculating, smoothing reduce the confidence (overfeed). ^this mean every high probability will have 0.1 taken and given to others.


    ## Traing loop ##
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"processing epoch {epoch:02d}") # this is the progress bar
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) # (B, Seq_Len)
            decoder_input = batch["decoder_input"].to(device) # (B, Seq_Len)
            encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, Seq_Len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, Seq_Len, Seq_Len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_Len, d_model)
            proj_output = model.project(decoder_output) # (B, Seq_Len, tgt_vocab_size)

            label = batch["label"].to(device) # (B, Seq_Len)

            # (B, Seq_Len, tgt_vocab_size) --> (B * Seq_Len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
            }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)