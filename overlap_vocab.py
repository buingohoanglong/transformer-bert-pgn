from transformers import AutoTokenizer
from Dictionary import Dictionary
from collections import Counter

with open('./data/vi-ba/dict_vi.txt', mode='r', encoding='utf-8') as file_vi:
    with open('./data/vi-ba/dict_ba.txt', mode='r', encoding='utf-8') as file_ba:
        src = file_vi.readlines()
        tgt = file_ba.readlines()
        src = [line[:line.find(" ")] for line in src]
        tgt = [line[:line.find(" ")] for line in tgt]
        overlap = list(set(src) & set(tgt))
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        dictionary = Dictionary(tokenizer=tokenizer)
        dictionary.add_from_file('./data/vi-ba/dict.txt')
        dictionary.build_dictionary()

        counter = Counter()
        for line in overlap:
            counter[line] = dictionary.counter[line]

        with open('./data/vi-ba/dict_overlap.txt', mode='w+', encoding='utf-8') as file_overlap:
            for token, count in counter.most_common():
                file_overlap.write(token + ' ' + count + '\n')