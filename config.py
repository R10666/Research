## This file is essentially the training loop setup ##
from pathlib import Path

def get_config():
    return {
        "batch_size": 8, # number of batch/traing example processed at once, so 8 sentence. 
        "num_epochs": 20, # each epochs is 1 complete pass through of the dataset, so runs dataset 20 times
        "lr": 10**-4, #Learning rate, determining the size of step taken in optimization. ##Note This can be dynamic and gradually changed throughout training.
        "seq_len": 350, # max sequence length for input. Rough and safe estimate = 350
        "d_model": 512, # diamention of hidden state, 512 from paper
        "lang_src": "en", 
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)