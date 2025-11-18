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

big_df = pd.read_csv("data/temp_stab_hand_scrap_row_seq.csv")
big_sequences = big_df['final_sequence'].to_list()
big_sequences_array = esmc_encode_batch(big_sequences)
seq_df = pd.DataFrame(big_sequences_array)

res = pd.concat([big_df, seq_df], axis=1)
res.to_pickle("data_embed/" + "temp_stab_hand_scrap_row_seq_esmc" + ".pkl")
print(f"DONE")
