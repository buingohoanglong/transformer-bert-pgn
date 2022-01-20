import torch
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

        
class NoneableTensorDataset(Dataset):
    def __init__(self, *tensors):
        assert len(tensors) > 0
        assert isinstance(tensors[0], torch.Tensor)
        assert all((isinstance(t, torch.Tensor) and tensors[0].shape[0] == t.shape[0]) or (t is None) for t in tensors)
        self.tensors = tensors

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return tuple(t[index].tolist() if t is not None else None for t in self.tensors)

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.tensors[0].shape[0]

    @staticmethod
    def collate_function(batch):
        new_batch = [[] for _ in range(len(batch[0]))]
        for sample in batch:
            for id, ele in enumerate(sample):
                new_batch[id].append(ele)
        return tuple(torch.as_tensor(batch_ele) if batch_ele[0] is not None else None for batch_ele in new_batch)