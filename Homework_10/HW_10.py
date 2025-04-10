from datasets import load_dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import accuracy_score

# --- Шаг 1: Загрузка датасета ---
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])


# --- Шаг 2: Очистка текста ---
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Удаляем HTML-теги
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
    text = text.lower()  # Приводим к нижнему регистру
    return text

train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["clean_text"], train_df["label"], test_size=0.2, random_state=42
)

# --- Шаг 3: Токенизация данных ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df["clean_text"]), truncation=True, padding=True, max_length=256)


# --- Шаг 4: Создание PyTorch Dataset ---
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_df["label"])

# --- Шаг 5: Fine-tuning модели BERT ---
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- Шаг 6: Оценка модели на тестовых данных ---
results = trainer.evaluate(test_dataset)
print(f"Точность на тестовом наборе: {results['eval_accuracy']:.4f}")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.argmax().item(), probs[0][1].item()

sample_text = "This movie was absolutely fantastic! I loved every minute of it."
class_label, pos_prob = predict_sentiment(sample_text)
print(f"\nПример предсказания:")
print(f"Текст: {sample_text}")
print(f"Класс: {'Положительный' if class_label == 1 else 'Отрицательный'}")
print(f"Вероятность положительного: {pos_prob:.4f}")