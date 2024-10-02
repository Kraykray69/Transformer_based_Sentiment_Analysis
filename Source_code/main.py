import mlflow
from data_preprocessing import prepare_data
from model_training import init_model, train_model, evaluate_model
from performance_monitoring import monitor_performance
from retraining import retrain_model
import torch

def main():
    # Start MLflow run
    with mlflow.start_run():
        # Prepare data
        train_dataloader, val_dataloader, tokenizer = prepare_data("C:\\Users\\rupes\\Desktop\\Problem_1_Rupesh\\Data\\IMDB Dataset.csv", sample_size=10000)
        
        # Initialize and train model
        model = init_model()
        model = train_model(model, train_dataloader, val_dataloader, epochs=1)
        
        # Evaluate the model
        final_accuracy = evaluate_model(model, val_dataloader)
        print(f"Final model accuracy: {final_accuracy:.4f}")
        
        # Log final metrics and model
        mlflow.log_metric("final_accuracy", final_accuracy)
        mlflow.pytorch.log_model(model, "model")
        
        # Save the model
        torch.save(model.state_dict(), "text_classification_model.pth")
        
        # Simulate performance degradation
        degraded_accuracy = 0.80  # This would normally come from monitoring live data
        if monitor_performance(degraded_accuracy):
            new_accuracy = retrain_model(model, train_dataloader, val_dataloader)
            print(f"Model retrained. New accuracy: {new_accuracy:.4f}")

if __name__ == "__main__":
    main()