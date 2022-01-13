from transformers import AutoTokenizer, AutoModel
from vncorenlp import VnCoreNLP
# from underthesea import ner
from Dictionary import preprocess
from utils import ner_for_bpe
import re
import unicodedata
from Dictionary import *
import torch

# annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word", use_fast=False)

bartpho = AutoModel.from_pretrained("vinai/bartpho-word")

text = 'Chúng_tôi là những nghiên_cứu_viên .'
ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(ids)
print(ids)
print(tokens)

features = bartpho(torch.tensor(ids).unsqueeze(0)).last_hidden_state.detach()
print(features.size())

_, preds = torch.max(features, dim=-1)
preds = preds.tolist()
print(preds)
out = tokenizer.convert_ids_to_tokens(preds[0])
print(out)

# # load dictionary
# dictionary = Dictionary(tokenizer=tokenizer)
# dictionary.add_from_file('./data/vi-ba/dict_pgn.txt')
# dictionary.build_dictionary()
# print(f'--|Vocab size: {len(dictionary)}')

# # text = "Ông Nguyễn Khắc Chúc 35 tuổi đang làm việc tại Đại học Quốc gia Hà Nội được 10 năm. Bà Lan 30 tuổi, vợ ông Chúc, cũng làm việc tại đây."
# text = "Chị gái của Yôl biết rất nhiều chuyện dân tộc Kinh."

# data = preprocess(annotator, text, ner=True)

# print(f'--|Words: {data["words"]}')
# print(f'--|Size: {len(data["words"])}\n')

# ne_tokens = data['name_entities']
# print(f'--|NER tokens: {ne_tokens}')
# print(f'--|Size: {len(ne_tokens)}\n')


# bpe_tokens = ["<s>"] + tokenizer.tokenize(" ".join(data['words'])) + ["</s>"]
# print(f'--|BPE tokens: {bpe_tokens}')
# print(f'--|Size: {len(bpe_tokens)}\n')


# ne_tokens_ext = ner_for_bpe(bpe_tokens, ne_tokens, get_mask=False, special_tokens=["<s>", "</s>"])
# print(f'--|NER extend tokens: {ne_tokens_ext}')
# print(f'--|Size: {len(ne_tokens_ext)}\n')


# ne_mask = ner_for_bpe(bpe_tokens, ne_tokens, get_mask=True, special_tokens=["<s>", "</s>"])
# print(f'--|NER mask: {ne_mask}')
# print(f'--|Size: {len(ne_mask)}\n')

# ids = dictionary.encode(" ".join(data['words']), append_eos=True)['ids']
# print(f'--|Ids: {ids}')
# print(f'--|Size: {len(ids)}\n')