import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import warnings

def load_model(model_path):
    global model, tokenizer
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def batch_inference(input_file, output_file, batch_size=32):
    df = pd.read_csv(input_file)
    texts = df['review'].tolist()
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        batch_predictions = torch.argmax(probabilities, dim=1).tolist()
        predictions.extend(batch_predictions)
    
    df['prediction'] = ['positive' if p == 1 else 'negative' for p in predictions]
    df.to_csv(output_file, index=False)
    print(f"Batch predictions saved to {output_file}")

# Suppress all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize model and tokenizer (ensure they are downloaded in your environment)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # This is a pre-trained model for sentiment analysis
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# HTML template for the main page with updated styles
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f4f9;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            max-width: 500px;
            width: 90%;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
            resize: none;
        }
        button {
            padding: 10px 20px;
            margin-top: 10px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        #result {
            margin-top: 15px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <form id="sentiment-form">
            <textarea id="text-input" placeholder="Enter text here..."></textarea>
            <br>
            <button type="submit">Analyze Sentiment</button>
        </form>
        <p id="result"></p>
    </div>

    <script>
        document.getElementById('sentiment-form').addEventListener('submit', function(e) {
            e.preventDefault();
            var text = document.getElementById('text-input').value;
            if (!text.trim()) {
                document.getElementById('result').textContent = 'Please enter some text.';
                return;
            }
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').textContent = 'Prediction: ' + data.prediction;
            })
            .catch(error => {
                document.getElementById('result').textContent = 'Error analyzing sentiment.';
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
"""

# Route to serve the main page
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Map prediction to sentiment
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify({'prediction': sentiment})

if __name__ == '__main__':
    print("Starting Flask app for live inference...")
    app.run(debug=True, use_reloader=False)
