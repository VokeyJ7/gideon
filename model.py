from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn

# Eyes
model = YOLO("yolov8n")

r = model.predict(source=0,show=True)

results_dict = r.names


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        assert embed_size % heads == 0, "Head size incompatible with Embed_size."
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, keys, values, query, mask):
        N = values.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = self.values(values).reshape(N, value_len, self.heads, self.heads_dim)
        keys = self.keys(keys).reshape(N, key_len, self.heads, self.heads_dim)
        queries = self.queries(query).reshape(N, query_len, self.heads, self.heads_dim)



        energy = torch.einsum("nqhd,nkhd->nhqk",queries, keys)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)

        
        attention = torch.softmax(energy / (self.heads_dim ** 0.5), dim=-1)
        out = torch.einsum("nhqk,nkhd->nqhd", attention, values).reshape(N, query_len, self.heads_dim*self.heads)
        out = self.fc_out(out)
        return out
        
# decoder-only infrastructure for chat-style LM
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
       )
    def forward(self, x, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        x = self.dropout(self.norm1(attention + x))
        fwd = self.fc(x)
        out = self.dropout(self.norm2(x + fwd))

        return out
    

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super().__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.device = device
    def make_trg_mask(self, trg, pad_idx):
        N, T = trg.shape
        causal = torch.tril(torch.ones(T, T, device=self.device)).bool()
        causal = causal.view(1,1,T,T).expand(N, 1, T, T)
        pad_ok = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
        return pad_ok & causal

    def forward(self, trg, pad_idx):
        N, seq_len = trg.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embeddings(trg) + self.position_embeddings(positions))
        trg_mask = self.make_trg_mask(trg, pad_idx)

        for layer in self.layers:
            x = layer(x, trg_mask)
        
        out = self.fc_out(x)
        return out
            
class Transformer(nn.Module):
    def __init__(self, trg_vocab_size, trg_pad_idx, embed_size=256, forward_expansion=4, num_layers=6, heads=8, dropout=0.1, device="cpu", max_length=100 ):
        super().__init__()
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.trg_pad_idx = trg_pad_idx
        self.device = device
  
    
    def forward(self, trg):
        trg = trg.to(self.device)
        out = self.decoder(trg, self.trg_pad_idx)
        return out
    


