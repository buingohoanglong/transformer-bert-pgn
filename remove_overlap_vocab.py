with open('./data/vi-ba/dict.txt', mode='r', encoding='utf-8') as file_vocab:
    with open('./data/vi-ba/dict_overlap.txt', mode='r', encoding='utf-8') as file_overlap:
        vocab = file_vocab.readlines()
        overlap = file_overlap.readlines()[1500:]
        vocab_new = [line for line in vocab if line not in overlap]
        with open('./data/vi-ba/dict_pgn.txt', mode='w+', encoding='utf-8') as file_vocab_new:
            file_vocab_new.writelines(vocab_new)