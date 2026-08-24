from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(
    os.environ.get("EXIST2026_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
REPO_ROOT = Path(os.environ.get("LNR_REPO_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()
DATASET_ROOT = Path(
    os.environ.get("EXIST2026_DATASET_ROOT", REPO_ROOT / "EXIST 2026 Dataset V0.2")
).expanduser().resolve()
if os.environ.get("EXIST2026_MEMES_ROOT"):
    MEMES_ROOT = Path(os.environ["EXIST2026_MEMES_ROOT"]).expanduser().resolve()
elif (DATASET_ROOT / "training" / "EXIST2026_training.json").exists():
    MEMES_ROOT = DATASET_ROOT
else:
    MEMES_ROOT = DATASET_ROOT / "EXIST 2026 Memes Dataset"
_DEFAULT_EVAL_ROOT = Path(
    os.environ.get("EXIST2026_EVAL_ROOT", DATASET_ROOT / "evaluation")
).expanduser().resolve()
if (
    not os.environ.get("EXIST2026_EVAL_ROOT")
    and not _DEFAULT_EVAL_ROOT.exists()
    and (REPO_ROOT / "evaluation").exists()
):
    EVAL_ROOT = (REPO_ROOT / "evaluation").resolve()
else:
    EVAL_ROOT = _DEFAULT_EVAL_ROOT

TRAIN_JSON = MEMES_ROOT / "training" / "EXIST2026_training.json"
TEST_JSON = MEMES_ROOT / "test" / "EXIST2026_test_clean.json"
TRAIN_IMAGES = MEMES_ROOT / "training" / "memes"
TEST_IMAGES = MEMES_ROOT / "test" / "memes"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_ROOT / "cache"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
MODELS_DIR = OUTPUT_ROOT / "models"
DEV_PRED_DIR = OUTPUT_ROOT / "dev_predictions"
TEAM_NAME = os.environ.get("EXIST2026_TEAM_NAME", "TEAM_NAME_NOT_SET")
RUNS_DIR = PROJECT_ROOT / "runs" / f"exist2026_{TEAM_NAME}"

SEED = 42
DEV_SIZE = 0.20
TEST_CASE = "EXIST2025"

TASKS = ("task2_1", "task2_2", "task2_3")
TASK_NAMES = {
    "task2_1": "Sexism identification in memes",
    "task2_2": "Source intention in memes",
    "task2_3": "Sexism categorization in memes",
}
LABELS = {
    "task2_1": ["NO", "YES"],
    "task2_2": ["NO", "JUDGEMENTAL", "DIRECT"],
    "task2_3": [
        "NO",
        "IDEOLOGICAL-INEQUALITY",
        "STEREOTYPING-DOMINANCE",
        "OBJECTIFICATION",
        "SEXUAL-VIOLENCE",
        "MISOGYNY-NON-SEXUAL-VIOLENCE",
    ],
}
FACET_LABELS = [label for label in LABELS["task2_3"] if label != "NO"]
GOLD_FILES = {
    "task2_1": EVAL_ROOT / "golds" / "EXIST2025_training_task2_1_gold_hard.json",
    "task2_2": EVAL_ROOT / "golds" / "EXIST2025_training_task2_2_gold_hard.json",
    "task2_3": EVAL_ROOT / "golds" / "EXIST2025_training_task2_3_gold_hard.json",
}
SOFT_GOLD_FILES = {
    "task2_1": EVAL_ROOT / "golds" / "EXIST2025_training_task2_1_gold_soft.json",
    "task2_2": EVAL_ROOT / "golds" / "EXIST2025_training_task2_2_gold_soft.json",
    "task2_3": EVAL_ROOT / "golds" / "EXIST2025_training_task2_3_gold_soft.json",
}


def ensure_dirs() -> None:
    for path in [
        CACHE_DIR,
        DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        DEV_PRED_DIR,
        RUNS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_meme_frame(split: str) -> pd.DataFrame:
    if split not in {"training", "test"}:
        raise ValueError("split must be 'training' or 'test'")
    json_path = TRAIN_JSON if split == "training" else TEST_JSON
    image_root = TRAIN_IMAGES if split == "training" else TEST_IMAGES
    raw = read_json(json_path)
    rows = []
    for fallback_id, sample in raw.items():
        meme_file = sample.get("meme") or Path(sample.get("path_memes", "")).name
        rows.append(
            {
                "id": str(sample.get("id_EXIST", fallback_id)),
                "lang": sample.get("lang"),
                "text": sample.get("text", ""),
                "meme": meme_file,
                "image_path": str(image_root / meme_file),
                "split": split,
                "source_split": sample.get("split", ""),
                "sensorial": sample.get("sensorial", {}),
            }
        )
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    return frame


def _label_above_threshold(labels: Iterable[str], valid_labels: Iterable[str], threshold: int) -> str | None:
    counts = Counter(label for label in labels if label in valid_labels)
    winners = [label for label, count in counts.items() if count > threshold]
    if len(winners) == 1:
        return winners[0]
    return None


def derive_hard_gold_from_training(task: str) -> pd.DataFrame:
    """Build hard labels from the meme training JSON when evaluation/golds is absent.

    The thresholds follow the EXIST 2026 guidelines for meme hard-hard evaluation:
    task2_1 uses labels with more than 3 annotators, task2_2 uses DIRECT/JUDGEMENTAL
    labels with more than 2 annotators, and task2_3 keeps sexism facets selected by
    more than 1 annotator. In tasks 2.2 and 2.3, a hard NO in task2_1 is propagated as
    NO to preserve the challenge hierarchy.
    """
    raw = read_json(TRAIN_JSON)
    rows = []
    for fallback_id, sample in raw.items():
        item_id = str(sample.get("id_EXIST", fallback_id))
        task21 = _label_above_threshold(sample.get("labels_task2_1", []), ["NO", "YES"], 3)

        if task == "task2_1":
            if task21 is not None:
                rows.append({"id": item_id, "gold": task21})
            continue

        if task == "task2_2":
            if task21 is None:
                continue
            if task21 == "NO":
                rows.append({"id": item_id, "gold": "NO"})
                continue
            labels22 = sample.get("labels_task2_2", [])
            counts22 = Counter(label for label in labels22 if label in ["JUDGEMENTAL", "DIRECT"])
            if counts22:
                max_count = max(counts22.values())
                if max_count > 2:
                    tied = {label for label, count in counts22.items() if count == max_count}
                    for label in labels22:
                        if label in tied:
                            rows.append({"id": item_id, "gold": label})
                            break
            continue

        if task == "task2_3":
            if task21 is None:
                continue
            if task21 == "NO":
                rows.append({"id": item_id, "gold": ["NO"]})
                continue
            counts: Counter[str] = Counter()
            for annotator_labels in sample.get("labels_task2_3", []):
                if not isinstance(annotator_labels, list):
                    continue
                for label in annotator_labels:
                    if label in FACET_LABELS:
                        counts[label] += 1
            facets = [label for label in FACET_LABELS if counts[label] > 1]
            if facets:
                rows.append({"id": item_id, "gold": facets})
            continue

        raise ValueError(f"Unknown task: {task}")

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def load_gold(task: str) -> pd.DataFrame:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if GOLD_FILES[task].exists():
        gold = pd.DataFrame(read_json(GOLD_FILES[task]))
        gold["id"] = gold["id"].astype(str)
        return gold[["id", "value"]].rename(columns={"value": "gold"})
    return derive_hard_gold_from_training(task)


def load_soft_gold(task: str) -> pd.DataFrame:
    """Load official soft labels as a two-column ``id``/``soft_gold`` frame."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    path = SOFT_GOLD_FILES[task]
    if not path.exists():
        raise FileNotFoundError(
            f"Soft-gold file not found: {path}. Set EXIST2026_EVAL_ROOT to the "
            "directory that contains the official evaluation/golds folder."
        )
    soft = pd.DataFrame(read_json(path))
    soft["id"] = soft["id"].astype(str)
    return soft[["id", "value"]].rename(columns={"value": "soft_gold"})


def attach_gold(records: pd.DataFrame, task: str) -> pd.DataFrame:
    gold = load_gold(task)
    merged = records.merge(gold, on="id", how="inner")
    if task == "task2_3":
        merged["gold"] = merged["gold"].apply(lambda x: x if isinstance(x, list) else [x])
    return merged.sort_values("id").reset_index(drop=True)


def attach_gold_and_soft(records: pd.DataFrame, task: str) -> pd.DataFrame:
    """Attach the official hard and soft labels used by the audited notebooks."""
    merged = attach_gold(records, task)
    return merged.merge(load_soft_gold(task), on="id", how="left").sort_values("id").reset_index(drop=True)


def text_inputs(frame: pd.DataFrame) -> pd.Series:
    """Return language-tagged OCR text with stable whitespace normalization."""
    text = frame.get("text", pd.Series("", index=frame.index)).fillna("").astype(str)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    lang = frame.get("lang", pd.Series("", index=frame.index)).fillna("").astype(str)
    return "[LANG_" + lang + "] " + text


def split_train_dev(task_frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task == "task2_3":
        stratify = task_frame.apply(
            lambda row: f"{row['lang']}_{'NO' if row['gold'] == ['NO'] else 'SEXIST'}", axis=1
        )
    else:
        stratify = task_frame["lang"].astype(str) + "_" + task_frame["gold"].astype(str)

    counts = Counter(stratify)
    stratify_values = stratify if min(counts.values()) >= 2 else None
    train_idx, dev_idx = train_test_split(
        np.arange(len(task_frame)),
        test_size=DEV_SIZE,
        random_state=SEED,
        stratify=stratify_values,
    )
    train = task_frame.iloc[train_idx].sort_values("id").reset_index(drop=True)
    dev = task_frame.iloc[dev_idx].sort_values("id").reset_index(drop=True)
    return train, dev


def labels_for_task(task: str) -> list[str]:
    return LABELS[task]


def encode_multilabel(values: Iterable[list[str]]) -> tuple[np.ndarray, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer(classes=LABELS["task2_3"])
    y = mlb.fit_transform(list(values))
    return y, mlb


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def aggregate_sensorial(sensorial: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    users = sensorial.get("users") or []
    features["sens_users_count"] = float(len(users))
    modalities = sensorial.get("modalities") or {}
    for modality in ("ET", "HR", "EEG"):
        by_user = ((modalities.get(modality) or {}).get("by_user") or {})
        features[f"sens_{modality}_users_count"] = float(len(by_user))
        values_by_feature: dict[str, list[float]] = defaultdict(list)
        for user_features in by_user.values():
            if not isinstance(user_features, dict):
                continue
            for key, value in user_features.items():
                numeric = _safe_float(value)
                if not math.isnan(numeric):
                    safe_key = (
                        key.replace(" ", "_")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("/", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )
                    values_by_feature[safe_key].append(numeric)
        for key, values in values_by_feature.items():
            arr = np.asarray(values, dtype=float)
            prefix = f"sens_{modality}_{key}"
            features[f"{prefix}_mean"] = float(np.nanmean(arr))
            features[f"{prefix}_std"] = float(np.nanstd(arr))
    return features


def _normalized_hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _entropy_from_hist(hist: np.ndarray) -> float:
    probs = hist[hist > 0]
    return float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0


def _lbp_hist(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    codes = np.zeros(center.shape, dtype=np.uint8)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy : gray.shape[0] - 1 + dy, 1 + dx : gray.shape[1] - 1 + dx]
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    return _normalized_hist(codes.ravel(), bins=bins, value_range=(0, 256))


def extract_visual_features(image_path: str | Path) -> dict[str, float]:
    path = Path(image_path)
    features: dict[str, float] = {}
    try:
        with Image.open(path) as raw_image:
            raw_image = ImageOps.exif_transpose(raw_image).convert("RGB")
            width, height = raw_image.size
            features["vis_width"] = float(width)
            features["vis_height"] = float(height)
            features["vis_aspect_ratio"] = float(width / height) if height else 0.0
            features["vis_log_area"] = float(np.log1p(width * height))
            features["vis_is_landscape"] = float(width >= height)

            image = ImageOps.fit(raw_image, (128, 128), method=Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    except Exception:
        features["vis_load_error"] = 1.0
        return features

    features["vis_load_error"] = 0.0
    channel_names = ("r", "g", "b")
    for idx, name in enumerate(channel_names):
        values = rgb[:, :, idx].ravel()
        features[f"vis_{name}_mean"] = float(values.mean())
        features[f"vis_{name}_std"] = float(values.std())
        features[f"vis_{name}_q10"] = float(np.quantile(values, 0.10))
        features[f"vis_{name}_q50"] = float(np.quantile(values, 0.50))
        features[f"vis_{name}_q90"] = float(np.quantile(values, 0.90))
        for bin_id, value in enumerate(_normalized_hist(values, bins=8, value_range=(0, 1))):
            features[f"vis_{name}_hist_{bin_id:02d}"] = float(value)

    for idx, name in enumerate(("h", "s", "v")):
        values = hsv[:, :, idx].ravel()
        features[f"vis_hsv_{name}_mean"] = float(values.mean())
        features[f"vis_hsv_{name}_std"] = float(values.std())
        for bin_id, value in enumerate(_normalized_hist(values, bins=8, value_range=(0, 1))):
            features[f"vis_hsv_{name}_hist_{bin_id:02d}"] = float(value)

    gray_values = gray.ravel()
    gray_hist = _normalized_hist(gray_values, bins=16, value_range=(0, 1))
    for bin_id, value in enumerate(gray_hist):
        features[f"vis_gray_hist_{bin_id:02d}"] = float(value)
    features["vis_gray_mean"] = float(gray_values.mean())
    features["vis_gray_std"] = float(gray_values.std())
    features["vis_gray_entropy"] = _entropy_from_hist(gray_hist)
    features["vis_dark_pixel_ratio"] = float((gray_values < 0.10).mean())
    features["vis_light_pixel_ratio"] = float((gray_values > 0.90).mean())

    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    features["vis_colorfulness"] = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    grad_y, grad_x = np.gradient(gray)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = (np.arctan2(grad_y, grad_x) + np.pi) / (2 * np.pi)
    features["vis_gradient_mean"] = float(magnitude.mean())
    features["vis_gradient_std"] = float(magnitude.std())
    features["vis_edge_density_005"] = float((magnitude > 0.05).mean())
    features["vis_edge_density_010"] = float((magnitude > 0.10).mean())
    orient_hist, _ = np.histogram(
        orientation.ravel(), bins=12, range=(0, 1), weights=magnitude.ravel()
    )
    orient_total = orient_hist.sum()
    if orient_total > 0:
        orient_hist = orient_hist / orient_total
    for bin_id, value in enumerate(orient_hist):
        features[f"vis_orientation_hist_{bin_id:02d}"] = float(value)

    for i, y_idx in enumerate(np.array_split(np.arange(gray.shape[0]), 4)):
        for j, x_idx in enumerate(np.array_split(np.arange(gray.shape[1]), 4)):
            block = gray[np.ix_(y_idx, x_idx)]
            block_mag = magnitude[np.ix_(y_idx, x_idx)]
            features[f"vis_grid_{i}{j}_gray_mean"] = float(block.mean())
            features[f"vis_grid_{i}{j}_gray_std"] = float(block.std())
            features[f"vis_grid_{i}{j}_edge_mean"] = float(block_mag.mean())

    for bin_id, value in enumerate(_lbp_hist(gray, bins=32)):
        features[f"vis_lbp_hist_{bin_id:02d}"] = float(value)

    return features


def build_feature_frame(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    cache_path = CACHE_DIR / "meme_features.csv"
    if cache_path.exists() and not force:
        records = pd.concat([load_meme_frame("training"), load_meme_frame("test")], ignore_index=True)
        cached = pd.read_csv(cache_path, dtype={"id": str})
        if {"id", "image_path"}.issubset(cached.columns) and len(cached) == len(records):
            expected_paths = records.set_index("id")["image_path"].astype(str)
            cached_paths = cached.set_index("id")["image_path"].astype(str)
            sample_ids = list(expected_paths.index[:20]) + list(expected_paths.index[-20:])
            if all(item_id in cached_paths.index and cached_paths.loc[item_id] == expected_paths.loc[item_id] for item_id in sample_ids):
                return cached
        print("Feature cache does not match the current dataset paths; recomputing it.")
    else:
        records = pd.concat([load_meme_frame("training"), load_meme_frame("test")], ignore_index=True)

    feature_rows = []
    for idx, row in records.iterrows():
        if idx and idx % 250 == 0:
            print(f"Extracted features for {idx}/{len(records)} memes")
        features = {
            "id": row["id"],
            "lang": row["lang"],
            "split": row["split"],
            "text": row["text"],
            "meme": row["meme"],
            "image_path": row["image_path"],
        }
        features.update(extract_visual_features(row["image_path"]))
        features.update(aggregate_sensorial(row["sensorial"]))
        feature_rows.append(features)

    frame = pd.DataFrame(feature_rows).sort_values("id").reset_index(drop=True)
    frame.to_csv(cache_path, index=False)
    return frame


def task_metric(task: str, y_true: Any, y_pred: Any) -> dict[str, float]:
    if task == "task2_1":
        labels = LABELS[task]
        return {
            "main_metric": float(f1_score(y_true, y_pred, pos_label="YES", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    if task == "task2_2":
        labels = LABELS[task]
        return {
            "main_metric": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    if task == "task2_3":
        mlb = MultiLabelBinarizer(classes=LABELS[task])
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        return {
            "main_metric": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
            "exact_match": float((y_true_bin == y_pred_bin).all(axis=1).mean()),
        }
    raise ValueError(f"Unknown task: {task}")


def per_label_f1(task: str, y_true: Any, y_pred: Any) -> pd.DataFrame:
    if task in {"task2_1", "task2_2"}:
        labels = LABELS[task]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        return pd.DataFrame(
            {
                "task": task,
                "label": labels,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    mlb = MultiLabelBinarizer(classes=LABELS[task])
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, zero_division=0
    )
    return pd.DataFrame(
        {
            "task": task,
            "label": LABELS[task],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def single_label_report(task: str, y_true: Any, y_pred: Any) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS[task],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in LABELS[task]:
        row = report[label]
        rows.append(
            {
                "task": task,
                "label": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": row["support"],
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(task: str, y_true: Any, y_pred: Any) -> pd.DataFrame:
    labels = LABELS[task]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            rows.append({"task": task, "gold": gold, "predicted": pred, "count": int(cm[i, j])})
    return pd.DataFrame(rows)


def normalize_task23_prediction(labels: list[str]) -> list[str]:
    clean = [label for label in labels if label in LABELS["task2_3"]]
    facets = [label for label in clean if label != "NO"]
    if facets:
        return facets
    return ["NO"]


def write_hard_run(task: str, ids: Iterable[str], predictions: Iterable[Any], path: Path) -> None:
    rows = []
    for item_id, pred in zip(ids, predictions):
        value = normalize_task23_prediction(pred) if task == "task2_3" else str(pred)
        rows.append({"test_case": TEST_CASE, "id": str(item_id), "value": value})
    write_json(rows, path)


def write_soft_run(task: str, ids: Iterable[str], probabilities: Iterable[Any], path: Path) -> None:
    labels = LABELS[task]
    rows = []
    for item_id, probs in zip(ids, probabilities):
        probs_arr = np.asarray(probs, dtype=float)
        value = {label: float(probs_arr[i]) for i, label in enumerate(labels)}
        rows.append({"test_case": TEST_CASE, "id": str(item_id), "value": value})
    write_json(rows, path)


def summarize_prediction_distribution(task: str, predictions: Iterable[Any]) -> pd.DataFrame:
    if task != "task2_3":
        counts = Counter(predictions)
        return pd.DataFrame(
            [{"task": task, "label": label, "count": counts.get(label, 0)} for label in LABELS[task]]
        )
    counts = Counter()
    for pred in predictions:
        for label in normalize_task23_prediction(pred):
            counts[label] += 1
    return pd.DataFrame(
        [{"task": task, "label": label, "count": counts.get(label, 0)} for label in LABELS[task]]
    )
