import torch
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
df = df[["v1", "v2"]]
df['label'] = df.v1.map({'ham': 0, 'spam': 1})
print(df)

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/bert-tiny-finetuned-sms-spam-detection")
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/bert-tiny-finetuned-sms-spam-detection", num_labels=2)

# test the model accuracy on sample data
sample_data = df['v2'].to_list()[4000:4100]
sample_labels = df['label'].to_list()[4000:4100]
token_sample_output = tokenizer(sample_data, padding=True, return_tensors="pt")
output_prediction = model(**token_sample_output)
numpy_output = output_prediction.logits.detach().numpy()
preds = np.argmax(numpy_output, axis=-1)
print(accuracy_score(sample_labels, preds))
# 0.97


# Save the model to a pickle file
filename = "smsmodel.pkl"
with open(filename, 'wb') as f:
    pickle.dump((model, tokenizer), f)


# from datasets import Dataset
# train_data_df = df.sample(200, random_state=42)
# eval_data_df = df.sample(200, random_state=45)
# train_dataset = Dataset.from_pandas(train_data_df)
# eval_dataset = Dataset.from_pandas(eval_data_df)
# test_data_df = df.iloc[4000:4100]
# test_dataset = Dataset.from_pandas(test_data_df)
#
#
# def tokenize_function(examples):
#     return tokenizer(examples["v2"], padding="max_length", truncation=True)
#
#
# tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_datasets_eval = eval_dataset.map(tokenize_function, batched=True)
# tokenizer_datasets_test = test_dataset.map(tokenize_function, batched=True)
#
# from transformers import TrainingArguments, Trainer
#
# training_args = TrainingArguments(output_dir="test_trainer")
#
# import numpy as np
# import evaluate
#
# metric = evaluate.load("accuracy")
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets_train,
#     eval_dataset=tokenized_datasets_eval,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# predictions_output = trainer.predict(tokenizer_datasets_test)
# accuracy_score = compute_metrics((predictions_output.predictions,tokenizer_datasets_test['label']))
# print(accuracy_score)
