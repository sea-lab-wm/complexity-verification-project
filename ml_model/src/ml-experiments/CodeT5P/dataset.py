from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
from rtdl_num_embeddings import LinearReLUEmbeddings

# Define a custom dataset class
class CodeDataset(Dataset):
    def __init__(self, tokenizer, csv_file, target=None, code_features=None):
        self.tokenizer = tokenizer
        self.target = target
        self.data = pd.read_csv(csv_file)
        self.code_features = code_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code_snippet = self.data['file_content'][idx]
        code_features_values = self.data[self.code_features].iloc[idx].values ## [1,1]
        # code_features_values = code_features_values.reshape(len(self.code_features), -1) ## reshape for the embedding layer

        target = self.data[self.target][idx]
        encoded_code_snippet = self.tokenizer(code_snippet, 
                                              return_tensors='pt', 
                                              padding='max_length', 
                                              truncation=True, 
                                              max_length=512)
        # code_feature_embedding_layer = LinearReLUEmbeddings(len(self.code_features), 2)
        # code_feature_embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=512)
        # ## Transform the continuous variables using the embedding instance
        # code_feature_embedding = code_feature_embedding_layer(torch.LongTensor(code_features_values)).squeeze()
        code_snippet_embedding = encoded_code_snippet['input_ids'].squeeze()
        code_snippet_attention_mask = encoded_code_snippet['attention_mask'].squeeze()
        code_feature_embedding=torch.LongTensor([[0, 2, 0, 5]])
        return code_snippet_embedding, code_feature_embedding, code_snippet_attention_mask, torch.tensor(target)