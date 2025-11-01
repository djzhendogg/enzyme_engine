#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
python3 esmc_embeddings.py > esmc_embeddings.log
python3 prott5_embeddings.py > prott5_embeddings.log
python3 prot_bert_embeddings.py > prot_bert_embeddings.log