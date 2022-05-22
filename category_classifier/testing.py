import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from tabulate import tabulate
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast

# loading finetune model & tokenizer
model_path = "question-classification-model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
print("model & tokenizer loaded...")

# create pipeline object
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# loading test data
df = pd.read_csv("../preprocessing/test.csv", low_memory=False, usecols=["Text", "Category"]).replace(np.nan, "None")
print(f"test data loaded...\n -shape: {df.shape}")

# inference
result = [nlp(text) for text in tqdm(df.Text)]
df["prediction"] = [res[0]["label"] for res in tqdm(result)]
df["score"] = [res[0]["score"] for res in tqdm(result)]
print("inference done...")

# classification report
report = metrics.classification_report(y_true=df.Category.tolist(), y_pred=df.prediction.tolist(), output_dict=True)
print(f"Accuracy: {round(report['accuracy'], 2)}", file=open("./classification_report.txt", "a"))
print(f"Macro Avg Precision: {round(report['macro avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))
print(f"Weighted Avg Precision: {round(report['weighted avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))

report = pd.DataFrame([
    {
        "label": key,
        "precision": value["precision"],
        "recall": value["recall"],
        "support": value["support"]
    }
    for key, value in report.items()
    if key not in ["accuracy", "macro avg", "weighted avg"]
])
print(
    tabulate(
        report,
        headers="keys",
        tablefmt="psql"
    ),
    file=open("./classification_report.txt", "a")
)