import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
from sklearn.decomposition import PCA
import joblib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cpu":
    exit(1)

model_name="esmc_600m"

def esmc_encode_batch(sequences, batch_size=16):
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            for seq in batch_seqs:
                prot = ESMProtein(sequence=seq)
                protein_tensor = client.encode(prot)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                emb = logits_output.embeddings.squeeze(0)  # (seq_len, embedding_dim)
                emb_mean = emb.mean(dim=0).cpu().numpy()
                embeddings.append(emb_mean)

    embeddings = np.vstack(embeddings)
    return embeddings

big_df = pd.read_csv("R02112_with_sequences.csv")
big_df.drop(['uniprot_sequence'], axis=1, inplace=True)
big_df.dropna(subset="ncbi_sequence", inplace=True)

big_sequences = big_df['ncbi_sequence'].to_list()
big_sequences_array = esmc_encode_batch(big_sequences)
df = pd.DataFrame(big_sequences_array)
df.to_pickle("big_esmc_embeddings.pkl")
pca = PCA(n_components=2)
components = pca.fit_transform(df)
components = pd.DataFrame(components, columns=['Principle Component 0', 'Principle Component 1'])
components.to_pickle("big_esmc_embeddings_PCA.pkl")
joblib.dump(pca, 'pca_big_esmc_embeddings_model.pkl')
