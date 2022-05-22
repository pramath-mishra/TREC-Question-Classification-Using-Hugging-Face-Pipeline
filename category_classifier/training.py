import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from torch import cuda
from tabulate import tabulate
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# availability of GPU
device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Device: {device}")

# loading train data
train = pd.read_csv(
    "../preprocessing/train.csv",
    low_memory=False,
    usecols=[
        "Text",
        "Category"
    ]
).replace(np.nan, "None")
print(f"train data loaded...\n -shape: {train.shape}")

# loading test data
test = pd.read_csv(
    "../preprocessing/test.csv",
    low_memory=False,
    usecols=[
        "Text",
        "Category"
    ]
).replace(np.nan, "None")
print(f"test data loaded...\n -shape: {test.shape}")

# label encoding mapping
labels = list(set(train.Category.tolist()))
NUM_LABELS = len(labels)
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}
print("label encoding mapping created...")

# train data label encoding
train["labels"] = train.Category.map(lambda x: label2id[x.strip()])
print(f"train data label encoded...\n -shape: {train.shape}")

# test data label encoding
test["labels"] = test.Category.map(lambda x: label2id[x.strip()])
print(f"test data label encoded...\n -shape: {test.shape}")

# loading pre-train BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)

# loading pre-train BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# preparing train & test data
train_texts = train.Text.tolist()
test_texts = test.Text.tolist()

train_labels = train.labels.tolist()
test_labels = test.labels.tolist()
print(f"train & test data prepared...\n -Train count: {len(train_texts)} Test count: {len(test_texts)}")

# train & test data encoding
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print("train & test data encoded...")


# preparing dataset
class MyDataset(Dataset):
    def __init__(self, encodings, label):
        self.encodings = encodings
        self.labels = label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MyDataset(train_encodings, train_labels)
test_dataset = MyDataset(test_encodings, test_labels)


# compute metrics
def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(label, preds, average='macro')
    acc = accuracy_score(label, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


# updating training arguments
training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./Model',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    # TensorBoard log directory
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    fp16=False if device == "cpu" else True,
    load_best_model_at_end=False
)
print("training arguments updated...")

# create trainer object
trainer = Trainer(
    # the pre-trained model that will be fine-tuned
    model=model,
    # training arguments that we defined above
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
print("trainer object created...")

# start model training
trainer.train()
print("model training done...")

# evaluation
q = [trainer.evaluate(eval_dataset=data) for data in [train_dataset, test_dataset]]
print(
    tabulate(
        pd.DataFrame(q, index=["train", "test"]).iloc[:, :5],
        headers="keys",
        tablefmt="psql"
    ),
    file=open("./evaluation_report.txt", "a")
)

# saving the fine-tuned model & tokenizer
model_path = "question-classification-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print("model saved...")
