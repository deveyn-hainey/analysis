"""SigLIP + UMAP + KMeans team classification, ported from Soccer_Analysis_Model.

HSV histograms confuse similar kits and shift with lighting; SigLIP visual
embeddings separate teams by everything visible on the player (kit pattern,
shorts, socks), which is what the offline model repo uses. Trained once per
clip on calibration crops, then applied per frame.

Fully optional: if transformers/umap-learn aren't importable, or training
fails (too few crops, degenerate clusters), callers fall back to the HSV path.
Enable/disable with YOLO_TEAM_BACKEND=siglip|hsv (default siglip).
"""
from typing import Optional

import numpy as np
from PIL import Image

from .config import logger

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
EMBED_BATCH_SIZE = 24
UMAP_COMPONENTS = 3
# Below this many calibration crops, UMAP/KMeans overfit noise — let the HSV
# path handle sparse clips instead.
MIN_TRAINING_CROPS = 16


class SiglipTeamClassifier:
    """Per-clip classifier: fit on calibration crops, predict per frame."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._reducer = None
        self._kmeans = None
        self.trained = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoProcessor, SiglipVisionModel

            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            self._model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME).to(self._device)
            self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            logger.info("siglip team classifier loaded on %s", self._device)
            return True
        except Exception as exc:
            logger.warning("siglip unavailable (%s) — falling back to HSV team clustering", exc)
            return False

    def _embed(self, crops: list[Image.Image]) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(crops), EMBED_BATCH_SIZE):
                batch = crops[start:start + EMBED_BATCH_SIZE]
                inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
                outputs = self._model(**inputs)
                chunks.append(torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def fit_predict(self, crops: list[Image.Image]) -> Optional[np.ndarray]:
        """Train UMAP + KMeans on calibration crops; return their cluster labels."""
        if len(crops) < MIN_TRAINING_CROPS or not self._ensure_model():
            return None
        try:
            import umap.umap_ as umap
            from sklearn.cluster import KMeans

            embeddings = self._embed(crops)
            self._reducer = umap.UMAP(n_components=UMAP_COMPONENTS)
            reduced = self._reducer.fit_transform(embeddings)
            self._kmeans = KMeans(n_clusters=2, n_init=10)
            labels = self._kmeans.fit_predict(reduced)
            if len(set(labels.tolist())) < 2:
                logger.warning("siglip clustering degenerate (one cluster) — falling back to HSV")
                return None
            self.trained = True
            logger.info(
                "siglip team classifier trained on %d crops (cluster sizes: %d / %d)",
                len(crops), int((labels == 0).sum()), int((labels == 1).sum()),
            )
            return labels
        except Exception as exc:
            logger.warning("siglip training failed (%s) — falling back to HSV", exc)
            return None

    def predict(self, crops: list[Image.Image]) -> Optional[np.ndarray]:
        if not self.trained or not crops:
            return None
        try:
            embeddings = self._embed(crops)
            reduced = self._reducer.transform(embeddings)
            return self._kmeans.predict(reduced)
        except Exception as exc:
            logger.debug("siglip predict failed for a frame (%s) — HSV fallback", exc)
            return None


def crops_for_detections(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> list[Image.Image]:
    crops = []
    for x1, y1, x2, y2 in boxes:
        crop = image.crop((int(x1), int(y1), int(max(x1 + 1, x2)), int(max(y1 + 1, y2))))
        crops.append(crop)
    return crops
