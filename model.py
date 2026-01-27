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
        assert(self.heads_dim*embed_size == heads), "Head size incompatible with Embed_size."
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.heads_dim, self.embed_size)
    
    def forward(self, keys, values, query, mask):
        N = values.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd nkhd -> nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("1e20"))

        
        attention = torch.softmax(energy / (self.heads_dim ** 0.5), dim=3)
        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads_dim*self.heads)
        out = self.fc_out(out)
        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, heads, embed_size, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(heads, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, keys, query, values, mask):
        attention = self.attention(keys, query, values, mask)
        x = self.dropout(self.norm1(attention + query))
        fwd = self.fc(x)
        out = self.dropout(self.norm2(fwd + x))
        return out
        
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super().__init__()
        self.device = device
        self.word_embeddings = nn.Embeddings(src_vocab_size, embed_size)
        self.position_embeddings = nn.Embeddings(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(heads, embed_size, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embeddings(x), self.position_embeddings(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.transformerBlock = TransformerBlock(heads, embed_size, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    def forward(self, x, keys, values, trg_mask, src_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm1(attention + x))
        out = self.transformerBlock(keys, query, values, src_mask)
        return out
    

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super().__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in num_layers
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embeddings(x) + self.position_embeddings(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, trg_mask, src_mask)
        out = self.fc_out(x)
        return out
            
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, trg_pad_idx, src_pad_idx, embed_size=256, forward_expansion=4, num_layers=6, heads=8, dropout=0, device="cpu", max_length=100 ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    


