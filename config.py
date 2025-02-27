## This file is essentially the training loop setup ##
from pathlib import Path

def get_config():
    return {
        "batch_size": 8, # number of batch/traing example processed at once, so 8 sentence. 
        "num_epochs": 25, # each epochs is 1 complete pass through of the dataset, so runs dataset 20 times
        "lr": 10**-4, #Learning rate, determining the size of step taken in optimization. ##Note This can be dynamic and gradually changed throughout training.
        "seq_len": 490, # max sequence length for input. Rough and safe estimate = 350
        "d_model": 512, # diamention of hidden state, 512 from paper
        "lang_src": "en", # source langauge
        "lang_tgt": "fr", # output langauge
        "model_folder": "en_fr_weights", # name of folder where each training/epochs is stored
        "model_basename": "tmodel_", # the name of file for each epoch
        #"preload": "35", # set a name of file to load and restart train if crash 
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json", # file for tokens
        "experiment_name": "runs/en-fr" # just a name for tensorboard logging
    }

#generate path and saves the weight files
def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

# def get_weights_file_path(config, epoch: str):
#     model_folder = f"{config['datasource']}_{config['model_folder']}"
#     model_filename = f"{config['model_basename']}{epoch}.pt"
#     return str(Path('.') / model_folder / model_filename)

# # Find the latest weights file in the weights folder
# def latest_weights_file_path(config):
#     model_folder = f"{config['datasource']}_{config['model_folder']}"
#     model_filename = f"{config['model_basename']}*"
#     weights_files = list(Path(model_folder).glob(model_filename))
#     if len(weights_files) == 0:
#         return None
#     weights_files.sort()
#     return str(weights_files[-1])