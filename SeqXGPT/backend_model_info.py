# All models that Sniffer will use.
en_model_names = ['gpt_2']
cn_model_names = ['wenzhong', 'sky_text', 'damo', 'chatglm']

# feature
tot_feat_num = 4
# Warn: when use 'loss_only' or 'feature_only', 
# you need change the hidden_size in both train.py and the backend_sniffer.py
train_feat = 'all'
cur_feat_num = 4
# checkpoint path, this is corresponding to the `cur_feat_num` and `train_feat`
en_ckpt_path = ""
cn_ckpt_path = ""

en_labels = {
        'gpt2': 0,
        'llama': 1,
        'human': 2,
        'gpt3re': 3,
    }
en_class_num = 4

cn_labels = {
    'wenzhong': 0,
    'sky_text': 1,
    'damo': 2,
    'chatglm': 3,
    'gpt3re': 4,
    'gpt3sum': 4,
    'human': 5
}
cn_class_num = 6

base_model = "roberta-base"
ckpt_name = ''.format('CNN')
