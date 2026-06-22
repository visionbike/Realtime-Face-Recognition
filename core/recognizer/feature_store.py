from pathlib import Path
import numpy as np


def read_features(path: str | Path) -> tuple[np.ndarray, np.ndarray] |  None:
    """
    Load saved face features from a .pnz file.

    :param path: Path to the feature file, without the ".npz" extension.
    :return: (image_names, image_embeddings) arrays, or None if the file is missing, unreadable, or does not contain the expected keys.
    """
    try:
        file_path = Path(path) / "features.npz"
        data = np.load(f"{file_path}", allow_pickle=True)
        image_name = data["image_name"]
        image_embeddings = data["image_embeddings"]
        return image_name, image_embeddings
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None


def compare_embeddings(embedding: np.ndarray, embeddings: np.ndarray) -> tuple[float, int]:
    """
    Find the closest match for an embedding within a set of embeddings.

    :param embedding: Query embedding of shape (1, D).
    :param embeddings: Stored embeddings of shape (N, D).
    :return:
        best_score: The best similarity score.
        best_idx: The index of the closest match.
    """
    similarity_scores = np.dot(embeddings, embedding.T)
    best_idx = int(np.argmax(similarity_scores))
    best_score = similarity_scores[best_idx].item()
    return best_score, best_idx
