"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from networks.transformer.model.decoder import Decoder
from networks.transformer.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)

        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * \
                   self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    device = 'cpu'
    b = 1
    max_len = 11
    encoder = Encoder(enc_voc_size=768, max_len=max_len, d_model=512, ffn_hidden=2048, n_head=8, n_layers=6, drop_prob=0.1, device=device)

    decoder = Decoder(dec_voc_size=10, max_len=3, d_model=512, ffn_hidden=2048, n_head=8, n_layers=6, drop_prob=0.1, device=device)

    id = torch.randint(1, 256, (b, max_len)).to(device)
    en = encoder(id, None)

    tg = torch.randint(1, 5, (b, 4)).to(device)
    tg_mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

    output = decoder(tg[:, :-1], en, tg_mask, None)
    # output = torch.softmax(output, dim=2)
    # output = torch.argmax(output, dim=2)
    output_reshape = output.contiguous().view(-1, output.shape[-1])
    trg = tg[:, 1:].contiguous().view(-1) # Class indices
    loss = criterion(output_reshape, trg)
    print(1)
