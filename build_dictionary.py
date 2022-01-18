from Dictionary import Dictionary, build_vocab
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

def main():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=9000) 
    dictionary = Dictionary(tokenizer=tokenizer)
    build_vocab(dictionary=dictionary, segmenter=rdrsegmenter, file_path='data/vi-ba/train.vi')
    build_vocab(dictionary=dictionary, segmenter=rdrsegmenter, file_path='data/vi-ba/train.ba')
    dictionary.build_dictionary()
    dictionary.save_vocabulary('./data/vi-ba/dict.txt')

if __name__ == '__main__':
    main()