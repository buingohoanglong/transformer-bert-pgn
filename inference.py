import torch

def inference(model, src, src_bert, max_sequence_length=256):
    """
    :param src: Tensor [batch_size, src_seq_len, d_model]
    :param src_bert: Tensor [batch_size, src_seq_len, d_model]
    :return tgt: Tensor [batch_size, max_seq_len]
    """
    tgt = torch.tensor([0]).repeat(src.size(0), 1) # [batch_size, current_len]
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    model.eval()
    for _ in range(max_sequence_length+1):
        output = model(src, tgt, src_bert)  # [batch_size, current_len, vocab_size]
        values, ids = torch.max(output, dim=-1)
        tgt = torch.cat([tgt,ids[:,-1].unsqueeze(1)],dim=-1)
    return tgt