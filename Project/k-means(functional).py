''' 
Acest cod este rezolvarea cu K-means pentru clustere, avand clasificare nesupervizata
Silhouette Score: 0.08159277468356585 care este unul destul de slab, neavand pe ce sa antrenam

Am vrut sa fac o implementare cu Large Language Model de text classification cu un model
open source de pe Hugging Face, insa nu il puteam rula pe local.

Adaug pe langa acest cod inca unul cu cel in care am incercat sa rulez cu LLM-ul.

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# citirea fisierelor csv
companies_df = pd.read_csv("ml_insurance_challenge.csv")  
taxonomy_df = pd.read_csv("insurance_taxonomy-insurance_taxonomy.csv")  

# Curatare nume coloane de eventuale spații ascunse
companies_df.columns = companies_df.columns.str.strip()
taxonomy_df.columns = taxonomy_df.columns.str.strip()

# Extragere etichete corecte
taxonomy_labels = taxonomy_df["label"].tolist()

# Combiare campuri pentru fiecare companie
companies_df["full_text"] = (
    companies_df["description"].fillna("") + " " +
    companies_df["business_tags"].fillna("") + " " +
    companies_df["sector"].fillna("") + " " +
    companies_df["category"].fillna("") + " " +
    companies_df["niche"].fillna("")
)

# Vectorizare 
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
X = tfidf.fit_transform(companies_df["full_text"])

# Aplicare KMeans pentru clustering
num_clusters = len(taxonomy_labels)  # Putem încerca și o selecție optimă
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Calcul scor silhouette pentru evaluare
sil_score = silhouette_score(X, clusters)
print(f"Silhouette Score: {sil_score}")

# Atribuire de etichete pe baza similaritatii cosine
taxonomy_vectors = tfidf.transform(taxonomy_labels)
similarity_matrix = cosine_similarity(X, taxonomy_vectors)
predicted_labels = [taxonomy_labels[np.argmax(sim)] for sim in similarity_matrix]

# Adaugare etichete 
companies_df["Predicted Label"] = predicted_labels

# Salvare rezultat
companies_df.to_csv("classified_companies.csv", index=False)
