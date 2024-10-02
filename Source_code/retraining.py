import torch
import mlflow
from model_training import train_model, evaluate_model

# Retrain function
def retrain_model(model, train_dataloader, val_dataloader, epochs=1):
    print("Time to retrain this bad boy...")
    mlflow.end_run()
    with mlflow.start_run():
        model = train_model(model, train_dataloader, val_dataloader, epochs=epochs)
        retrained_accuracy = evaluate_model(model, val_dataloader)
        print("Retrained model accuracy:", retrained_accuracy)
        mlflow.log_metric("retrained_accuracy", retrained_accuracy)
        mlflow.pytorch.log_model(model, "retrained_model")
        torch.save(model.state_dict(), "text_classification_model_retrained.pth")
    return retrained_accuracy