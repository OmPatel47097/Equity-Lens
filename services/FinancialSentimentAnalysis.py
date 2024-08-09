import os
import torch
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinancialSentimentAnalyzer:
    def __init__(
        self,
        model_name="mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
        model_dir="./saved_model",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        if os.path.exists(model_dir):
            print(f"Loading model from {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir
            ).to(self.device)
            mlflow.log_param("model_source", "local")
        else:
            print(f"Downloading model {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
            self.save_model()
            mlflow.log_param("model_source", "downloaded")

        mlflow.pytorch.log_model(self.model, "sentiment_model")

    def save_model(self):
        print(f"Saving model to {self.model_dir}")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        print(f"Model saved to {self.model_dir}")

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = probabilities[0][2].item() - probabilities[0][0].item()

        if sentiment_score > 0.2:
            return "Positive", sentiment_score
        elif sentiment_score < -0.2:
            return "Negative", sentiment_score
        else:
            return "Neutral", sentiment_score


        mlflow.log_metric("sentiment_score", sentiment_score)
        mlflow.log_param("sentiment", sentiment)
