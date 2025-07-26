import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm
from transformers import TFAutoModel, AutoTokenizer
from Bio.Align import substitution_matrices
from sklearn.decomposition import PCA

model_PB_name="Rostlab/prot_bert"
tokenizer_PB = AutoTokenizer.from_pretrained(model_PB_name, do_lower_case=False)
model_PB = TFAutoModel.from_pretrained(model_PB_name, from_pt=True)
blosum62 = substitution_matrices.load('BLOSUM62')
amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'

def prot_bert_encode_batch(sequences, device="GPU", batch_size=16):
    sequences = [" ".join(list(seq)) for seq in sequences]

    inputs = tokenizer_PB(sequences, return_tensors="tf", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = []
    with tf.device(f"/{device}:0"):
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding batches"):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]

            outputs = model_PB(input_ids=batch_input_ids, attention_mask=batch_attention_mask, training=False)
            hidden_states = outputs.last_hidden_state

            batch_embeddings = tf.reduce_mean(hidden_states[:, 1:-1, :], axis=1).numpy()
            embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

big_df = pd.read_csv("R02112_with_sequences.csv")
big_df.drop(['uniprot_sequence'], axis=1, inplace=True)
big_df.dropna(subset="ncbi_sequence", inplace=True)

big_sequences = big_df['ncbi_sequence'].to_list()
big_sequences_array = prot_bert_encode_batch(big_sequences)
df = pd.DataFrame(big_sequences_array)
df.to_csv("big_prot_bert_embeddings.csv")
pca = PCA(n_components=2)
components = pca.fit_transform(df)
components = pd.DataFrame(components, columns=['Principle Component 0', 'Principle Component 1'])
components.to_csv("big_prot_bert_embeddings_PCA.csv")
joblib.dump(pca, 'pca_big_prot_bert_embeddings_model.pkl')
