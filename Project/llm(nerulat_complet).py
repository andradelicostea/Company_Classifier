'''
Incercare cu large Language Model, insa nu am reusit sa-l rulez local si sa creez documentul.
Am facut prin auto-tokenizare pentru includerea modelui de pe Hugging Face, imbinare coloane pentru intrare.
apoi clasificare pe baza combinatiilor impartite in tokeni.
'''

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")

# Load datasets
companies_df = pd.read_csv("ml_insurance_challenge.csv")  # Replace with actual file
taxonomy_df = pd.read_csv("insurance_taxonomy-insurance_taxonomy.csv")  # Replace with actual file


companies_df.columns = companies_df.columns.str.strip()
taxonomy_df.columns = taxonomy_df.columns.str.strip()


taxonomy_labels = taxonomy_df["label"].tolist()

companies_df["combined_text"] = (
    companies_df["description"].fillna("") + " " +
    companies_df["business_tags"].fillna("") + " " +
    companies_df["sector"].fillna("") + " " +
    companies_df["category"].fillna("") + " " +
    companies_df["niche"].fillna("")
)

def classify_text(text):
    inputs = tokenizer(
        [[text, label] for label in taxonomy_labels],
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)[:, 1]  

    best_index = scores.argmax().item()
    best_label = taxonomy_labels[best_index]
    
    return best_label, scores[best_index].item()

companies_df["predicted_label"] = companies_df["combined_text"].apply(lambda x: classify_text(x)[0])

companies_df.to_csv("classified_companies2.csv", index=False)

print(companies_df[["combined_text", "predicted_label"]].head())
