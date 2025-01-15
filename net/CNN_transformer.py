import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self, num_classes1):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(13, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.fc = nn.Linear(768, num_classes1)


    def forward(self, x):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        print("head_dim:", self.head_dim)
        print("num_heads:", self.num_heads)
        print("embed_dim:", self.embed_dim)
        assert self.head_dim * num_heads == self.embed_dim


        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)


        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src):
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])


    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


# 定义包含 Transformer 的音频分类模型
class AudioTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, dim_feedforward):
        super(AudioTransformerModel, self).__init__()
        self.cnn = CNN(416) 
        encoder_layer = TransformerEncoderLayer(input_dim, num_heads, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(input_dim, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.cnn(x)  
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1).contiguous()
        x = nn.MaxPool2d(kernel_size=1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
