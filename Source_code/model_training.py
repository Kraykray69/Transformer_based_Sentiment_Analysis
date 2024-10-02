import torch
from transformers import DistilBertForSequenceClassification, AdamW
from tqdm import tqdm
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        val_accuracy = evaluate_model(model, val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
    
    return model

def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
    
    return correct_predictions / total_predictions

def init_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
    return model