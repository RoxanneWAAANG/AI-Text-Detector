import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm


def access_api(text, api_url, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        # msgpack.packb: Serializes post_data for transmission.
        # Sends a POST request to the API with the serialized data.
        # prediction = client.post(api_url,
        #                          data=msgpack.packb(post_data),
        #                          headers={"Content-Type": "application/msgpack"},
        #                          timeout=None)
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        # msgpack.unpackb: Deserializes the response.
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_features_unlabeled(input_file, output_file):
    """
    Extract features from unlabeled text using GPT-2 server.

    Args:
        input_file (str): Path to the input JSONL file containing only text.
        output_file (str): Path to save the extracted features.
    """

    # Define the API for GPT-2 model (update if needed)
    gpt_2_api = 'http://localhost:20098/inference'
    model_api = gpt_2_api  # Using only GPT-2 for inference

    # Read input file containing text samples
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    print(f'Processing {len(lines)} samples from {input_file}')

    # Open the output file to write processed features
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']

            # Initialize storage variables
            losses = []
            begin_idx_list = []
            ll_tokens_list = []

            # Call API to get model inference results
            try:
                loss, begin_word_idx, ll_tokens = access_api(line, model_api)
            except TypeError:
                print("API returned NoneType, possible GPU OOM. Skipping sample.")
                continue  # Skip faulty samples

            # Store extracted features
            losses.append(loss)
            begin_idx_list.append(begin_word_idx)
            ll_tokens_list.append(ll_tokens)

            # Save the extracted features
            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'text': line  # Retain the original text for reference
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Feature extraction complete. Results saved to {output_file}")
