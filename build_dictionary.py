from Dictionary import Dictionary, build_vocab
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

def main():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    dictionary = Dictionary(tokenizer=tokenizer)
    build_vocab(dictionary=dictionary, segmenter=rdrsegmenter, file_path='./data/vi-ba/train.vi')
    build_vocab(dictionary=dictionary, segmenter=rdrsegmenter, file_path='./data/vi-ba/train.ba')
    dictionary.build_dictionary()
    dictionary.save_vocabulary('./data/vi-ba/dict.txt')

if __name__ == '__main__':
    main()