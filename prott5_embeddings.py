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

            embeddings.append(emb_batch)

        embeddings = np.vstack(embeddings)

    return embeddings

big_df = pd.read_csv("R02112_with_sequences.csv")
big_df.drop(['uniprot_sequence'], axis=1, inplace=True)
big_df.dropna(subset="ncbi_sequence", inplace=True)

big_sequences = big_df['ncbi_sequence'].to_list()
big_sequences_1 = big_sequences[:big_sequences//2]
big_sequences_2 = big_sequences[big_sequences//2:]
big_sequences_array = prot_t5_encode_batch(big_sequences_1)
df = pd.DataFrame(big_sequences_array)
df.to_pickle("big_prott5_embeddings_1.pkl")
del df
del big_sequences_array

big_sequences_array = prot_t5_encode_batch(big_sequences_2)
df = pd.DataFrame(big_sequences_array)
df.to_pickle("big_prott5_embeddings_2.pkl")

del df
del big_sequences_array

df_1 = pd.read_pickle("big_prott5_embeddings_1.pkl")
df_2= pd.read_pickle("big_prott5_embeddings_2.pkl")
df = pd.concat([df_1, df_2])
pca = PCA(n_components=2)
components = pca.fit_transform(df)
components = pd.DataFrame(components, columns=['Principle Component 0', 'Principle Component 1'])
components.to_pickle("big_prott5_embeddings_PCA.pkl")
joblib.dump(pca, 'pca_big_prott5_embeddings_model.pkl')
