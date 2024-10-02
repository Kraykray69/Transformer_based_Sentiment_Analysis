import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(file_path, sample_size=None):
    df = pd.read_csv(file_path)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

def preprocess_data(df):
    texts = df['review'].tolist()
    labels = (df['sentiment'] == 'positive').astype(int).tolist()
    return texts, labels

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def create_dataloaders(train_texts, val_texts, train_labels, val_labels, tokenizer, batch_size=16):
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader

def prepare_data(file_path, sample_size=None, test_size=0.2, batch_size=16):
    data = load_data(file_path, sample_size)
    texts, labels = preprocess_data(data)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size, random_state=42)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataloader, val_dataloader = create_dataloaders(train_texts, val_texts, train_labels, val_labels, tokenizer, batch_size)
    
    return train_dataloader, val_dataloader, tokenizer