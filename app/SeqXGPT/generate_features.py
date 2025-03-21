import torch
import transformers
import re
import numpy as np
import json
from tqdm import tqdm
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# T5 FUNCTIONS FOR TEXT PERTURBATION
def tokenize_and_mask(text, span_length=2, pct=0.1, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")])
        for text in texts
    ]

def replace_masks(texts, tokenizer, model, mask_top_p=0.95, device='cuda'):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**tokens,
                             max_length=512,
                             do_sample=True,
                             top_p=mask_top_p,
                             num_return_sequences=1,
                             eos_token_id=stop_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts(text, tokenizer, model, ptb_nums, span_length=2, pct=0.3, device='cuda'):
    texts = [text for i in range(0, ptb_nums)]
    masked_texts = [tokenize_and_mask(x, span_length, pct) for x in texts]
    raw_fills = replace_masks(masked_texts, tokenizer, model, device=device)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    perturbed_texts = [text for text in perturbed_texts if text != '']
    return perturbed_texts

# FEATURE EXTRACTION FUNCTIONS
def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
    if chinese_char_count / total_char_count > cn_percent:
        return _split_cn_sentence(sentence, use_sp)
    else:
        return _split_en_sentence(sentence, use_sp)

def _split_en_sentence(sentence, use_sp=False):
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words

def _split_cn_sentence(sentence, use_sp=False):
    words = list(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words

class BBPETokenizerPPLCalc:
    def __init__(self, byte_encoder, model, tokenizer, device):
        self.byte_encoder = byte_encoder
        self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_bbpe_bytes(self, words):
        bbs = []  # bbpe_bytes
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits.squeeze()  # seq-len, V
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def get_bbs_ll(self, input_ids, ll):
        input_ids = input_ids.squeeze()
        tokenized_tokens = [
            self.tokenizer._convert_id_to_token(input_id)
            for input_id in input_ids
        ]
        bbs_ll = []
        # First token handling
        byte_list = [self.byte_decoder[c] for c in tokenized_tokens[0]]
        bbs_ll.extend([0 for i in range(len(byte_list))])
        # Following tokens
        for idx, token in enumerate(tokenized_tokens[1:]):
            byte_list = [self.byte_decoder[c] for c in token]
            bbs_ll.extend(ll[idx] for i in range(len(byte_list)))
        return bbs_ll

    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_token = bbs_ll[start:end]
            ll_tokens.append(np.mean(ll_token).item())
            start = end
        return ll_tokens

    def get_begin_word_idx(self, input_ids, bbs_to_words):
        input_ids = input_ids.squeeze()
        begin_token = self.tokenizer._convert_id_to_token(input_ids[0])
        byte_list = [self.byte_decoder[c] for c in begin_token]
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024]
        labels = labels[:, :1024]
        words = split_sentence(text)
        bbs, bbs_to_words = self.get_bbpe_bytes(words)

        outputs = self.model(input_ids=input_ids, labels=labels)
        loss, ll = self.calc_sent_ppl(outputs, labels)
        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(bbs_to_words, bbs_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens]

class FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load T5 model for text perturbation
        print("Loading T5 model...")
        self.t5_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
        self.t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('t5-base')
        self.t5_model.to(self.device)
        
        # Load GPT-2 model for perplexity calculation
        print("Loading GPT-2 model...")
        self.gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-medium')
        self.gpt2_model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-medium')
        self.gpt2_tokenizer.pad_token_id = self.gpt2_tokenizer.eos_token_id
        self.gpt2_model.to(self.device)
        
        # Initialize bytes encoder for tokenizer
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder, 
                                                  self.gpt2_model, 
                                                  self.gpt2_tokenizer, 
                                                  self.device)
    
    def extract_features(self, text):
        # Generate perturbed texts
        perturbed_texts = perturb_texts(text, self.t5_tokenizer, self.t5_model, 
                                        ptb_nums=10, device=self.device)
        
        # Calculate perplexity for original text
        self.gpt2_tokenizer.padding_side = 'right'
        ppl_results = self.ppl_calculator.forward_calc_ppl(text)
        loss, begin_word_idx, ll_tokens = ppl_results
        
        # Calculate perplexity for perturbed texts
        perturbed_losses = []
        for perturbed_text in perturbed_texts:
            try:
                perturbed_ppl = self.ppl_calculator.forward_calc_ppl(perturbed_text)
                perturbed_losses.append(perturbed_ppl[0])  # Just collect the loss
            except Exception as e:
                print(f"Error processing perturbed text: {e}")
                continue
        
        # Calculate feature differences
        perturbed_loss_mean = np.mean(perturbed_losses) if perturbed_losses else 0
        perturbed_loss_std = np.std(perturbed_losses) if perturbed_losses else 0
        
        # Return features in a format similar to the expected output
        features = {
            "text": text,
            "ll_tokens_list": ll_tokens
        }
        
        return features

def process_file(input_file, output_file, extractor, limit=None):
    """Process a JSONL file containing text samples."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        processed_count = 0
        for line in tqdm(f_in):
            if limit and processed_count >= limit:
                break
                
            sample = json.loads(line)
            text = sample.get('text', '')
            
            # Process the text
            features = extractor.extract_features(text)
            # Add any additional fields from the original sample
            for key, value in sample.items():
                if key != 'text':
                    features[key] = value
                    
            # Write to output file
            f_out.write(json.dumps(features) + '\n')
            processed_count += 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract features from text samples')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    extractor = FeatureExtractor(device=args.device)
    process_file(args.input_file, args.output_file, extractor, limit=args.limit)

if __name__ == "__main__":
    main()