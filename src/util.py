"""General utility function"""
from __future__ import division, print_function, unicode_literals

import json
import logging
import joblib
import os


# Loading model
def load_model(path_model):
    model = joblib.load(path_model)
    return model


# Saving model
def save_model(model, path_output):
    """Saving any model to PKL file
    Args:
        model: (object)
        path_output: (string) path of output
    """

    joblib.dump(model, path_output)
    logging.info('Model saved to %s' % (path_output))


# check directory if exist else create directory
def check_dir(path_output):
    """Checking directory and creating
    folder if doesn't exist
    Args:
        path_output: (string) directory
    """
    if not os.path.exists(path_output):
        os.makedirs(path_output)


# Loading json
def load_json(path_model):
    """Loads json object to dict
    Args:
        path_model: (string) path of input
    """
    with open(path_model) as f:
        data = json.load(f)
    return data


# Saving json
def save_json(model, path_output):
    """Saves dictionary to json object.
    
    Args:
        model: (Dict object)
        path_output: (string) path of output
    """
    # Saving keyword and keyword atc
    with open(path_output, 'w') as f:
        json.dump(model, f)


# Saving vocab to txt file
def save2txt(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to
    the id f the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (string) path to vocab file
    """
    with open(txt_path, 'w') as f:
        for token in vocab:
            f.write(str(token) + '\n')


# Loading txt as list
def load_txt(path_txt):
    with open(path_txt, encoding='utf-8') as f:
        vocab = f.read().splitlines()
    return vocab


# Setting logging
def set_logger(path_log):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(path_log)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s : %(levelname)s : %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)