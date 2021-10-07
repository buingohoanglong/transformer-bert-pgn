from torch.utils.data import Dataset

class NMTDataset(Dataset):
    def __init__(self, src_path, tgt_path):
        with open(src_path, mode='r', encoding='utf-8') as src_file:
            with open(tgt_path, mode='r', encoding='utf-8') as tgt_file:
                self.src = src_file.readlines()
                self.tgt = tgt_file.readlines()
                assert len(self.src) == len(self.tgt)
                self.num_samples = len(self.src)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.num_samples