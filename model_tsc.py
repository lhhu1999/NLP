import torch
import torch.nn as nn
import math
from transformer_encoder import Encoder


class Embeddings_add_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_add_position, self).__init__()
        self.emb_size = emb_size
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_dp):
        seq_length = input_dp.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_dp.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_dp).unsqueeze(2)

        words_embeddings = self.word_embeddings(input_dp)

        pe = torch.zeros(words_embeddings.shape).cuda()
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().cuda()
        pe[..., 0::2] = torch.sin(position_ids * div)
        pe[..., 1::2] = torch.cos(position_ids * div)

        embeddings = words_embeddings + pe
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TSC(nn.Module):
    def __init__(self, args, len_dict):
        super(TSC, self).__init__()
        self.word_dim = args.word_dim

        self.embedding = Embeddings_add_position(len_dict, self.word_dim)
        self.N = 3
        self.encoder = Encoder(self.word_dim, 2, 1536)
        self.predict = nn.Sequential(
            # nn.Linear(self.word_dim, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 2),
            # nn.Softmax(1)

            nn.Linear(self.word_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(1)
        )

    def forward(self, batch_words_idx, batch_words_mask):
        words_emb = self.embedding(batch_words_idx)

        words_emb = self.encoder(words_emb, batch_words_mask)
        words_f = torch.amax(words_emb, dim=1)
        out = self.predict(words_f)
        return out

