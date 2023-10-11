import torch.nn as nn
import torch.nn.functional as F
from transformer_encoder import Encoder


class NER(nn.Module):
    def __init__(self, bert_model):
        super(NER, self).__init__()
        self.bert_model = bert_model
        self.conv1 = nn.Conv1d(768, 768, kernel_size=(3,), padding=1)
        self.norm1 = nn.LayerNorm(768)
        self.lstm1 = nn.LSTM(768, 768, num_layers=2)

        self.encoder1 = Encoder(768, 3, 1536)
        self.encoder2 = Encoder(768, 3, 1536)
        self.encoder3 = Encoder(768, 3, 1536)

        # self.lstm2 = nn.LSTM(768, 2, num_layers=2)

        self.predict = nn.Sequential(
            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, batch_words_idx, batch_words_mask):
        words_f = self.bert_model(input_ids=batch_words_idx, attention_mask=batch_words_mask).last_hidden_state
        words_conv = self.conv1(words_f.permute(0, 2, 1)).permute(0, 2, 1)
        words_conv = F.relu(self.norm1(words_conv))
        words_lstm, _ = self.lstm1(words_conv)
        words_out = self.norm1(words_conv + words_lstm)

        words_out = self.encoder1(words_out, batch_words_mask)
        words_out = self.encoder2(words_out, batch_words_mask)
        words_out = self.encoder3(words_out, batch_words_mask)

        # out, _ = self.lstm2(words_out)
        out = self.predict(words_out)
        out = F.softmax(out, dim=2).view(-1, 2)
        return out
