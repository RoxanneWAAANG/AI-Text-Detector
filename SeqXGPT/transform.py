import torch
import numpy as np
import re


def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    """Splits a sentence into tokens based on language characteristics."""
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
    if chinese_char_count / total_char_count > cn_percent:
        return _split_cn_sentence(sentence, use_sp)
    else:
        return _split_en_sentence(sentence, use_sp)

def _split_en_sentence(sentence, use_sp=False):
    """Split an English sentence into tokens."""
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words

def _split_cn_sentence(sentence, use_sp=False):
    """Split a Chinese sentence into characters."""
    words = list(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def transform_inference_sample(sample, max_len=1024, prompt_len=0, device=None):
    """
    Transform a sample with extracted features into model inputs.
    
    Args:
        sample: Dict containing 'text' and 'll_tokens_list'
        max_len: Maximum sequence length to use
        prompt_len: Length of prompt in sample (optional)
        device: Device to place tensors on
    
    Returns:
        Dict with 'features' and 'labels' tensors ready for model input
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract text and features
    text = sample['text']
    ll_tokens = sample['ll_tokens_list']
    
    # For models trained with multiple language model features, we need to 
    # prepare the feature tensor correctly
    if isinstance(ll_tokens[0], list):
        # Already in the right format - multiple model features
        features = ll_tokens
    else:
        # Single list of token perplexities - needs to be converted to 4D format
        # Assumed to be from a single model, typically GPT-2
        features = [ll_tokens]
    
    # Determine how many tokens are in the text
    tokens = split_sentence(text)
    text_len = min(len(tokens), max_len)
    
    # Convert features to tensor and handle padding/truncation
    features_array = np.array(features).transpose()  # Shape: [seq_len, num_features]
    
    # Handle padding or truncation
    if features_array.shape[0] < max_len:
        # Pad
        padding = np.zeros((max_len - features_array.shape[0], features_array.shape[1]))
        features_array = np.vstack([features_array, padding])
    else:
        # Truncate
        features_array = features_array[:max_len]
    
    # Create feature tensor
    features_tensor = torch.tensor(features_array, dtype=torch.float32, device=device)
    
    # Create dummy labels tensor (-1 for padding, 0 for valid tokens)
    # This is used for mask generation during inference
    labels = torch.ones(max_len, dtype=torch.long, device=device) * -1
    labels[:text_len] = 0  # Set valid token positions to 0 (will be ignored during inference)
    
    # Add batch dimension
    features_tensor = features_tensor.unsqueeze(0)  # [1, seq_len, num_features]
    labels = labels.unsqueeze(0)  # [1, seq_len]
    
    return {
        'features': features_tensor,
        'labels': labels,
        'text_len': text_len
    }


def transform_batch_samples(samples, max_len=1024, device=None):
    """
    Transform a batch of samples into model inputs.
    
    Args:
        samples: List of dicts, each containing 'text' and 'll_tokens_list'
        max_len: Maximum sequence length to use
        device: Device to place tensors on
    
    Returns:
        Dict with batched 'features' and 'labels' tensors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_features = []
    batch_labels = []
    batch_text_lens = []
    
    for sample in samples:
        transformed = transform_inference_sample(sample, max_len, device=device)
        batch_features.append(transformed['features'])
        batch_labels.append(transformed['labels'])
        batch_text_lens.append(transformed['text_len'])
    
    # Stack tensors along batch dimension
    features_tensor = torch.cat(batch_features, dim=0)
    labels_tensor = torch.cat(batch_labels, dim=0)
    
    return {
        'features': features_tensor,
        'labels': labels_tensor,
        'text_lens': batch_text_lens
    }


def predict_with_model(model, sample, id2label, max_len=1024, device=None):
    """
    Make a prediction using a trained model.
    
    Args:
        model: The TransformerOnlyClassifier model
        sample: Dict containing 'text' and 'll_tokens_list'
        id2label: Dict mapping from label ID to label name
        max_len: Maximum sequence length
        device: Device to use
    
    Returns:
        Dict with prediction results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    inputs = transform_inference_sample(sample, max_len, device=device)
    
    with torch.no_grad():
        output = model(inputs['features'], inputs['labels'])
    
    # Get predicted path
    predictions = output['preds'][0].cpu().numpy()
    
    # Filter out padding (-1)
    valid_predictions = predictions[:inputs['text_len']]
    
    # Convert to labels
    pred_labels = [id2label.get(p, "unknown") for p in valid_predictions]
    
    # Calculate the most common predicted label
    from collections import Counter
    label_counts = Counter(pred_labels)
    most_common_label = label_counts.most_common(1)[0][0]
    
    return {
        'token_predictions': pred_labels,
        'text_prediction': most_common_label,
        'token_logits': output['logits'][0].cpu().numpy()
    }


if __name__ == "__main__":
    # Example usage
    sample = {
        'text': 'this is a test',
        'll_tokens_list': [0.0, 1.656, 1.656, 2.062, 2.062, 6.135, 6.135]
    }
    
    # For testing transformation only
    inputs = transform_inference_sample(sample)
    print(f"Features shape: {inputs['features'].shape}")
    print(f"Labels shape: {inputs['labels'].shape}")
    
    # For testing with a model (uncomment to use)
    """
    from model import TransformerOnlyClassifier
    
    # Initialize the model with the same parameters used during training
    id2label = {0: 'gpt2', 1: 'llama', 2: 'human', 3: 'gpt3re'}
    model = TransformerOnlyClassifier(id2labels=id2label, seq_len=1024)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('checkpoint/transformer_cls_model.pt'))
    
    # Make prediction
    result = predict_with_model(model, sample, id2label)
    print(f"Text prediction: {result['text_prediction']}")
    print(f"Token predictions (first 10): {result['token_predictions'][:10]}")
    """