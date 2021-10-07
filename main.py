from model import *
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from vncorenlp import VnCoreNLP
from Dictionary import *
from Dataset import *
from torch.utils.data import DataLoader
from NoamLRScheduler import *
from train import *

def main():
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")

    # load dictionary
    dictionary = Dictionary(tokenizer=tokenizer)
    dictionary.add_from_file('./data/dict.vi')
    dictionary.build_dictionary()
    print(f'--|Vocab size: {len(dictionary)}')

    # load dataset, dataloader
    dataset = NMTDataset('./data/train.vi', './data/train.en')
    print(f'--|Number of samples: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)

    # load model
    model = Transformer(
        vocab_size=len(dictionary),
        d_model=512, 
        d_ff=2048,
        num_heads=8, 
        num_layers=6, 
        dropout=0.1,
        bert=phobert,
        d_bert=768, 
        padding_idx=dictionary.token_to_index(dictionary.pad_token),
        use_pgn=True,
        unk_idx=dictionary.token_to_index(dictionary.unk_token)
    )
    print(model)

    # init learning rate, criterion, optimizer
    init_lr = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = NoamLRScheduler(optimizer, warmup_steps=4000, d_model=512)

    # train model
    train(model, criterion, optimizer, lr_scheduler, dataloader, dictionary, tokenizer, rdrsegmenter,
        num_epochs=5, accumulation_factor=512, use_pgn=True
    )



if __name__ == "__main__":
    main()