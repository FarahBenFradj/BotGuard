import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.score(lstm_out), dim=1)  # (B, L, 1)
        context = (weights * lstm_out).sum(dim=1)             # (B, H*2)
        return context, weights.squeeze(-1)


class BotDetector(nn.Module):
    """
    BiLSTM + Attention bot-generated comment detector.

    Text branch  : Embedding → BiLSTM(×2) → Attention → 128-dim
    Meta branch  : Dense(32) → Dense(16)              →  16-dim
    Fusion       : Concat → Dense(64) → Dropout → Sigmoid
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64,
                 meta_dim=10, dropout=0.3):
        super().__init__()

        # Text branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm    = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.text_bn   = nn.BatchNorm1d(hidden_dim * 2)

        # Metadata branch
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),       nn.ReLU()
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, text, meta):
        x = self.embedding(text)
        x, _ = self.bilstm(x)
        x, attn_w = self.attention(x)
        x = self.text_bn(x)
        m = self.meta_net(meta)
        out = self.classifier(torch.cat([x, m], dim=1))
        return out.squeeze(1), attn_w