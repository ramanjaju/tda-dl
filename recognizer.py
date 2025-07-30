import numpy as np
import pandas as pd
import os

EMBEDDING_FILE = "data/embeddings.csv"

def save_user_embedding(username, embeddings):
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = np.array(mean_embedding).flatten()  # <-- Force to 1D
    df = pd.DataFrame([[username, mean_embedding.tolist()]], columns=["name", "embedding"])
    header = not os.path.exists(EMBEDDING_FILE)
    df.to_csv(EMBEDDING_FILE, mode='a', header=header, index=False)


def load_embeddings():
    if not os.path.exists(EMBEDDING_FILE):
        return [], []
    df = pd.read_csv(EMBEDDING_FILE)
    names = df["name"].tolist()
    vectors = [np.array(eval(e)) for e in df["embedding"]]
    return names, vectors

def knn_match(embedding, k=1, threshold=0.75):
    from sklearn.neighbors import KNeighborsClassifier
    names, vectors = load_embeddings()
    if not vectors:
        return None
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(vectors, names)
    prob = knn.predict_proba([embedding])
    pred = knn.predict([embedding])[0]
    if max(prob[0]) >= threshold:
        return pred
    return None

