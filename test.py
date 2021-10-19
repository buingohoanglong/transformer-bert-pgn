import torch.nn as nn
from utils import *
from Dictionary import *
from model import NMT
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from Dataset import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")

    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file('./data/vi-ba/dict.txt')
    dictionary.build_dictionary(limit=8000)
    print(f'--|Vocab size: {len(dictionary)}')

    # load test_dataset, test_dataloader
    test_dataset = NMTDataset('./data/vi-ba/test.vi', './data/vi-ba/test.ba')
    print(f'--|Number of train samples: {len(test_dataset)}')
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=4)

    # init criterion
    criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token_to_index(dictionary.pad_token))

    # load model
    model = NMT.load_from_checkpoint(
        checkpoint_path="./checkpoints/last.ckpt",
        dictionary=dictionary, 
        tokenizer=tokenizer, 
        segmenter=rdrsegmenter, 
        criterion=criterion,
        d_model=512, 
        d_ff=2048,
        num_heads=8, 
        num_layers=6, 
        dropout=0.1,
        bert=phobert,
        d_bert=768,
        use_pgn=True,
        max_src_len=256,
        max_tgt_len=256
    )
    # print(model)
    
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0
    )

    # train
    trainer.test(model=model, dataloaders=test_loader)

    
if __name__ == '__main__':
    main()