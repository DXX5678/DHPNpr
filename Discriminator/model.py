import torch
import torch.nn as nn


class DiscriminatorShare(nn.Module):
    """
        构建判别器：（共享生成器中的编码器）

        参数:
        * embedding_dim 词嵌入维度
        * hidden_dim 隐藏层维度
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim=2):
        super(DiscriminatorShare, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, buggy_embedded, patch_embedded):
        lstm_out, _ = self.lstm(torch.cat((buggy_embedded, patch_embedded), dim=1))
        pooled_output = torch.max(lstm_out, 1)[0]
        output = self.drop(pooled_output)
        return self.fc(output)


class DiscriminatorNoShare(nn.Module):
    """
        构建判别器：（独立编码）

        参数:
        * vocab_size 词表大小
        * embedding_dim 词嵌入维度
        * hidden_dim 隐藏层维度
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2):
        super(DiscriminatorNoShare, self).__init__()
        self.buggy_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.patch_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, buggy_input, patch_input):
        buggy_embedded = self.buggy_embedding(buggy_input)
        patch_embedded = self.patch_embedding(patch_input)
        lstm_out, _ = self.lstm(torch.cat((buggy_embedded, patch_embedded), dim=1))
        pooled_output = torch.max(lstm_out, 1)[0]
        output = self.drop(pooled_output)
        return self.fc(output)
