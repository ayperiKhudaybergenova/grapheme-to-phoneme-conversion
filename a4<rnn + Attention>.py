# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

file_path = "nlp2-a4.train"
pairs = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        graphemes, phonemes = line.strip().split("\t")
        src_seq = list(graphemes)              
        tgt_seq = phonemes.split(" ")         
        pairs.append((src_seq, tgt_seq))

print(f"Loaded {len(pairs)} pairs, sample:", pairs[:3])

# %%
def build_vocab(seqs, specials=["<pad>", "<sos>", "<eos>"]):
    counter = Counter([tok for seq in seqs for tok in seq])
    vocab = specials + sorted(counter.keys())
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    return vocab, stoi, itos

src_vocab, src_stoi, src_itos = build_vocab([p[0] for p in pairs])
tgt_vocab, tgt_stoi, tgt_itos = build_vocab([p[1] for p in pairs])

PAD_IDX = src_stoi["<pad>"]
SOS_IDX = tgt_stoi["<sos>"]
EOS_IDX = tgt_stoi["<eos>"]

INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
print(f"Src vocab size: {INPUT_DIM}, Tgt vocab size: {OUTPUT_DIM}")

# %%

def encode_src(seq):  # graphemes
    return [src_stoi[c] for c in seq]

def encode_tgt(seq):  # phonemes
    return [SOS_IDX] + [tgt_stoi[c] for c in seq] + [EOS_IDX]

encoded_pairs = [(encode_src(src), encode_tgt(tgt)) for src, tgt in pairs]

# %%
class G2PDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_max = max(len(s) for s in src_seqs)
    tgt_max = max(len(t) for t in tgt_seqs)

    src_padded = [s + [PAD_IDX]*(src_max - len(s)) for s in src_seqs]
    tgt_padded = [t + [PAD_IDX]*(tgt_max - len(t)) for t in tgt_seqs]

    return torch.tensor(src_padded), torch.tensor(tgt_padded)

dataset = G2PDataset(encoded_pairs)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1)
        src_len = encoder_outputs.size(1)
        hidden_repeat = hidden.expand(-1, src_len, -1)
        energy = torch.tanh(self.attn(torch.cat((hidden_repeat, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

# %%
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim*2, output_dim)
        self.attention = attention
    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.uniform_(m.weight, -0.08, 0.08)
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

# %%
EMB_DIM = 128
HID_DIM = 256
LR = 0.0003
EPOCHS = 10
CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attn)

enc.apply(init_weights)
dec.apply(init_weights)
enc, dec = enc.to(DEVICE), dec.to(DEVICE)

enc_opt = optim.Adam(enc.parameters(), lr=LR)
dec_opt = optim.Adam(dec.parameters(), lr=LR)

print(f"Training on {DEVICE}...")

for epoch in range(EPOCHS):
    enc.train()
    dec.train()
    epoch_loss = 0

    for src, tgt in train_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        if tgt.size(1) <= 1:  # skip empty
            continue

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        encoder_outputs, hidden = enc(src)
        input_token = torch.tensor([SOS_IDX] * src.size(0), device=DEVICE)

        loss = 0
        seq_len = tgt.size(1)
        for t in range(1, seq_len):
            output, hidden = dec(input_token, hidden, encoder_outputs)
            loss += F.cross_entropy(output, tgt[:, t], ignore_index=PAD_IDX)
            input_token = tgt[:, t]  # Teacher forcing

        loss = loss / (seq_len - 1)
        if torch.isnan(loss):
            print("[Warning] NaN detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), CLIP)

        enc_opt.step()
        dec_opt.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")


# %%
def predict(word, enc, dec, src_stoi, tgt_itos, max_len=50):
    enc.eval()
    dec.eval()
    
    src_ids = [src_stoi[c] for c in word]  
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE) 
    
    with torch.no_grad():
        encoder_outputs, hidden = enc(src_tensor)
    
    input_token = torch.tensor([SOS_IDX], device=DEVICE)
    result = []
    
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = dec(input_token, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
        
        if pred_token == tgt_stoi["<eos>"]:
            break
        result.append(tgt_itos[pred_token])
        
        input_token = torch.tensor([pred_token], device=DEVICE)
    
    return " ".join(result)


# %%
output_file = "g2p_predictions.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for src_seq, _ in pairs: 
        word = "".join(src_seq) 
        pred_phonemes = predict(word, enc, dec, src_stoi, tgt_itos)
        f.write(f"{word}\t{pred_phonemes}\n")

print(f"Predictions saved to {output_file}")



