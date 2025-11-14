import re
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
import pandas as pd
from sklearn.decomposition import PCA
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cpu":
    exit(1)

model_name="Rostlab/prot_t5_xl_half_uniref50-enc"
# Загрузка токенизатора и модели
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_name).to(device)

def prot_t5_encode_batch(sequences, batch_size=16):
    # Предобработка последовательностей: замена редких аминокислот и разделение пробелами
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Токенизация с добавлением паддинга до длины самой длинной последовательности в батче
            encoded = tokenizer(batch_seqs, add_special_tokens=True, return_tensors="pt", padding="longest")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # shape (batch_size, seq_len, hidden_dim)

            # Усредняем по длине последовательности для каждого белка
            emb_batch = last_hidden.mean(dim=1)  # shape (batch_size, hidden_dim)

            embeddings.append(emb_batch.cpu().numpy())

        embeddings = np.vstack(embeddings)

    return embeddings

big_df = pd.read_csv("data/new_termostab_512.csv")
sequences = big_df['final_sequence'].to_list()

sequences_array = prot_t5_encode_batch(sequences)
seq_df = pd.DataFrame(sequences_array)
res = pd.concat([big_df, seq_df], axis=1)
res.to_pickle("data_embed/" + "new_termostab_512_prott5" + ".pkl")
print(f"DONE")
