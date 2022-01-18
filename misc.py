from transformers import AutoTokenizer, AutoModel
from vncorenlp import VnCoreNLP
from utils import ner_for_bpe, preprocess
from Dictionary import *

annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

phobert = AutoModel.from_pretrained("vinai/phobert-base")

# load dictionary
dictionary = Dictionary(tokenizer=tokenizer)
dictionary.add_from_file('./data/vi-ba/dict.txt')
dictionary.build_dictionary()
print(f'--|Vocab size: {len(dictionary)}')

# text = "Ông Nguyễn Khắc Chúc 35 tuổi đang làm việc tại Đại học Quốc gia Hà Nội được 10 năm. Bà Lan 30 tuổi, vợ ông Chúc, cũng làm việc tại đây."
# text = "Chị gái của Yôl biết rất nhiều chuyện dân tộc Kinh."
text = "Căn cứ Thông tư liên tịch số 13/2014/TTLT-BYT-BNNPTNT-BCT ngày 09 tháng 4 năm 2014 của Bộ Y tế, Bộ Nông nghiệp và Phát triển nông thôn và Bộ Công Thương về việc chỉ dẫn phân công, kết hợp trong quản lý quốc gia về an toàn thực phẩm;"

data = preprocess(annotator, text, ner=True)

print(f'--|Words: {data["words"]}')
print(f'--|Size: {len(data["words"])}\n')

ne_tokens = data['name_entities']
print(f'--|NER tokens: {ne_tokens}')
print(f'--|Size: {len(ne_tokens)}\n')


bpe_tokens = ["<s>"] + tokenizer.tokenize(" ".join(data['words'])) + ["</s>"]
print(f'--|BPE tokens: {bpe_tokens}')
print(f'--|Size: {len(bpe_tokens)}\n')


ne_tokens_ext = ner_for_bpe(bpe_tokens, ne_tokens, get_mask=False, special_tokens=["<s>", "</s>"])
print(f'--|NER extend tokens: {ne_tokens_ext}')
print(f'--|Size: {len(ne_tokens_ext)}\n')


ne_mask = ner_for_bpe(bpe_tokens, ne_tokens, get_mask=True, special_tokens=["<s>", "</s>"])
print(f'--|NER mask: {ne_mask}')
print(f'--|Size: {len(ne_mask)}\n')

ids = dictionary.encode(" ".join(data['words']), append_eos=True)['ids']
print(f'--|Ids: {ids}')
print(f'--|Size: {len(ids)}\n')