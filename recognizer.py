import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Directory where each user’s embeddings are stored separately
DATA_DIR = "embeddings"
def save_user_embedding(name, embeddings):
    """
    Save multiple embeddings for a user into a pickle file.
    Each user has their own file named <username>.pkl.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{name}.pkl")

    # Normalize and flatten all embeddings before saving
    processed = []
    for e in embeddings:
        e = np.array(e).flatten()
        norm = np.linalg.norm(e)
        if norm == 0:
            continue
        processed.append(e / norm)

    if len(processed) == 0:
        print(f"[WARN] No valid embeddings for user: {name}")
        return

    with open(file_path, "wb") as f:
        pickle.dump(processed, f)

    print(f"[INFO] Saved {len(processed)} embeddings for user '{name}'.")


def load_all_embeddings():
    """
    Load all embeddings and corresponding labels from the embeddings directory.
    """
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        return None, None, None

    all_embeddings = []
    all_labels = []

    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith(".pkl"):
            continue

        user_name = file_name.replace(".pkl", "")
        file_path = os.path.join(DATA_DIR, file_name)

        with open(file_path, "rb") as f:
            stored_embeddings = pickle.load(f)
            all_embeddings.extend(stored_embeddings)
            all_labels.extend([user_name] * len(stored_embeddings))

    if not all_embeddings:
        return None, None, None

    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(all_labels)

    return np.array(all_embeddings), all_labels_encoded, le


def knn_match(embedding, distance_threshold=0.6):
    """
    Match a given face embedding using a k-NN classifier with a distance threshold.
    """
    X, y, le = load_all_embeddings()
    if X is None:
        return None, 0.0

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X, y)

    distances, indices = knn.kneighbors([embedding])
    distance = distances[0][0]

    if distance <= distance_threshold:
        matched_label = knn.predict([embedding])[0]
        matched_user = le.inverse_transform([matched_label])[0]
        similarity = 1 - distance  # Convert distance to similarity
        return matched_user, similarity
    else:
        return None, 1 - distance


def svm_match(embedding, probability_threshold=0.7):
    """
    Match a given face embedding using a Support Vector Machine (SVM) classifier.
    """
    X, y, le = load_all_embeddings()
    if X is None:
        return None, 0.0

    svm = SVC(kernel='linear', probability=True, C=1.0)
    svm.fit(X, y)

    probabilities = svm.predict_proba([embedding])[0]
    max_prob = np.max(probabilities)

    if max_prob >= probability_threshold:
        matched_label = svm.predict([embedding])[0]
        matched_user = le.inverse_transform([matched_label])[0]
        return matched_user, max_prob
    else:
        return None, max_prob

