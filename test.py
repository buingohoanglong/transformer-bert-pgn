from Loss import Loss
from utils import *
from Dictionary import *
from model import NMT
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from Dataset import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def main():
    annotator = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=False)
    bert = AutoModel.from_pretrained("bert-base-multilingual-cased")

    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file('./data/vi-ba/dict.txt')
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # load test_dataset, test_dataloader
    test_dataset = NMTDataset('./data/vi-ba/test.vi', './data/vi-ba/test.ba')
    print(f'--|Number of train samples: {len(test_dataset)}')
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=4)

    # init criterion
    # criterion = nn.CrossEntropyLoss(ignore_index=dictionary.token_to_index(dictionary.pad_token))
    criterion = Loss(ignore_idx=dictionary.token_to_index(dictionary.pad_token), smoothing=0.1)

    # load model
    model = NMT.load_from_checkpoint(
        checkpoint_path="./checkpoints/last.ckpt",
        dictionary=dictionary, 
        bert_tokenizer=bert_tokenizer, 
        annotator=annotator, 
        criterion=criterion,
        d_model=512, 
        d_ff=2048,
        num_heads=8, 
        num_layers=6, 
        dropout=0.1,
        bert=bert,
        d_bert=768,
        use_pgn=True,
        use_ner=True,
        max_src_len=512,
        max_tgt_len=512
    )
    # print(model)
    
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0
    )

    # train
    trainer.test(model=model, dataloaders=test_loader)

    
if __name__ == '__main__':
    main()