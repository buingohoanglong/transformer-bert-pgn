from collections import Counter

class Dictionary():
    def __init__(self, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', tokenizer=None):
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.tokenizer = tokenizer

        self._reset()

    def _reset(self):
        self.token2index = {}
        self.index2token = {}
        self.counter = Counter()
        self.vocab_size = 4

        self.token2index[self.bos_token] = 0
        self.token2index[self.pad_token] = 1
        self.token2index[self.eos_token] = 2
        self.token2index[self.unk_token] = 3
        
        self.index2token = {v: k for k, v in self.token2index.items()}

    def __len__(self):
        return self.vocab_size

    def add_token(self, token):
        if self._is_special_token(token):
            return
        if token in self.counter:
            self.counter[token] += 1
        else:
            self.counter[token] = 1

    def index_to_token(self, index):
        if index < self.vocab_size:
            return self.index2token[index]
        return self.unk_token

    def token_to_index(self, token, update=False):
        if token in self.token2index:
            return self.token2index[token]
        if update:
            self.token2index[token] = self.vocab_size
            self.index2token[self.vocab_size] = token
            self.vocab_size += 1
            return self.token2index[token]
        else:
            return self.token2index[self.unk_token]

    def encode(self, text, append_bos=False, append_eos=False, update=False):
        tokens = self.tokenize(text)
        tokens = [self.bos_token] + tokens if append_bos else tokens
        tokens = tokens + [self.eos_token] if append_eos else tokens
        ids = [self.token_to_index(t, update=update) for t in tokens]
        return {
            'bpe_tokens': tokens,
            'ids': ids
        }

    def decode(self, ids):
        tokens = [self.index_to_token(idx) for idx in ids]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def save_vocabulary(self, file_path, limit=None):
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, count in self.counter.most_common(limit):
                f.write(token + " " + str(count) + "\n")

    def add_from_file(self, file_path, limit=None):
        self._reset()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_tmp in lines:
                line = line_tmp.strip()
                idx = line.rfind(" ")
                if idx == -1:
                    raise ValueError("Incorrect dictionary format, expected '<token> <count>'")
                token = line[:idx]
                count = line[idx+1:]
                self.counter[token] = count
                if limit and len(self.counter) > limit:
                    break
    
    def build_dictionary(self, limit=None):
        for token, count in self.counter.most_common(limit):
            self.token2index[token] = self.vocab_size
            self.index2token[self.vocab_size] = token
            self.vocab_size += 1

    def _is_special_token(self, token):
        return token in [self.bos_token, self.pad_token, self.eos_token, self.unk_token]


    def tokenize(self, text):
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        else:
            return text.strip().split(" ")

def preprocess(annotator, text, ner=False):
    text = text.replace('\xa0', ' ').strip()
    sentences = annotator.ner(text) if ner else annotator.tokenize(text)
    segments = []
    for s in sentences:
        segments.extend(s)
    if len(segments) == 0:
        return {'words': [], 'name_entities': []} if ner else {'words': []}
    if ner:
        words, name_entities = zip(*segments)
        return {
            'words': words,
            'name_entities': name_entities
        }
    else:
        return {'words': segments}


def build_vocab(dictionary, segmenter, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_tmp in lines:
            line = " ".join(preprocess(segmenter, line_tmp.strip())['words'])
            tokens = dictionary.tokenize(line)
            for t in tokens:
                dictionary.add_token(t)
