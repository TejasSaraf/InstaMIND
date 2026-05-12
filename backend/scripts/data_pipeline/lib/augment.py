

"""
Augmentation-based oversampling for minority classes.

Key design decisions:
  - Augmented frames are saved to disk (AUGMENTED/{config_hash}/...) so they
    exist as real files for the training data loader.
  - Output dir is versioned by config fingerprint — config changes produce
    a new directory, preventing silent reuse of stale augmented data.
  - Filenames include a deterministic hash of (source + index + seed)
    → reproducible, collision-free, safe to re-run.
  - Idempotency: existing augmented files are skipped, not overwritten.
  - Applied ONLY to training set, NEVER to val/test.
"""



from __future__ import annotations

import hashlib

import numpy as np

import pandas as pd

from PIL import Image, ImageEnhance, ImageFilter

from pathlib import Path



from .config import PipelineConfig

from .paths import AUGMENTED





def augment_image(img: Image.Image, rng: np.random.Generator,

                  cfg: PipelineConfig) -> Image.Image:

    """
    Apply stochastic augmentations to a PIL image.

    Transforms are mild enough to preserve semantic content while creating
    genuinely different training samples. Each augmentation is applied
    independently with its own probability.
    """

    br = cfg.aug_brightness_range

    cr = cfg.aug_contrast_range





    if rng.random() > 0.3:

        factor = (1.0 - br) + rng.random() * (2 * br)

        img = ImageEnhance.Brightness(img).enhance(factor)





    if rng.random() > 0.3:

        factor = (1.0 - cr) + rng.random() * (2 * cr)

        img = ImageEnhance.Contrast(img).enhance(factor)





    if rng.random() > 0.5:

        angle = rng.uniform(-cfg.aug_rotation_deg, cfg.aug_rotation_deg)

        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))





    if rng.random() < cfg.aug_flip_prob:

        img = img.transpose(Image.FLIP_LEFT_RIGHT)





    if rng.random() < cfg.aug_blur_prob:

        img = img.filter(ImageFilter.GaussianBlur(radius=cfg.aug_blur_radius))



    return img





def _deterministic_uid(stem: str, index: int, seed: int) -> str:

    """
    Generate a reproducible 8-char hex ID from source filename + index + seed.
    Same inputs always produce the same ID → safe re-runs.
    Different inputs produce different IDs → no collisions.
    """

    raw = f"{stem}_{index}_{seed}".encode("utf-8")

    return hashlib.md5(raw).hexdigest()[:8]





def oversample_with_augmentation(

    group: pd.DataFrame,

    target: int,

    rng: np.random.Generator,

    cfg: PipelineConfig,

) -> tuple[pd.DataFrame, int]:

    """
    Oversample minority class by generating augmented copies on disk.

    Returns:
      - DataFrame with original + augmented frame rows
      - Number of augmented files generated (for logging)

    Idempotency: if an augmented file already exists with the same name,
    it is reused (not re-generated or overwritten).
    """

    original = group.copy()

    n_needed = target - len(original)

    if n_needed <= 0:

        return original, 0



    aug_rows = []

    source_indices = rng.choice(len(original), size=n_needed, replace=True)

    generated = 0



    for i, src_idx in enumerate(source_indices):

        row = original.iloc[src_idx]

        src_path = Path(row["frame_path"])



        if not src_path.exists():

            continue





        config_hash = cfg.fingerprint()

        uid = _deterministic_uid(src_path.stem, i, cfg.random_seed)

        cls_dir = AUGMENTED / config_hash / row["split"] / row["incident_type"]

        cls_dir.mkdir(parents=True, exist_ok=True)

        aug_path = cls_dir / f"{src_path.stem}_aug_{uid}.jpg"





        if not aug_path.exists():

            try:

                img = Image.open(src_path).convert("RGB")

                aug_img = augment_image(img, rng, cfg)

                aug_img.save(str(aug_path), "JPEG", quality=cfg.jpeg_quality)

                generated += 1

            except Exception:

                continue

        else:

            generated += 0



        new_row = row.copy()

        new_row["frame_path"] = str(aug_path)

        aug_rows.append(new_row)



    if aug_rows:

        aug_df = pd.DataFrame(aug_rows)

        return pd.concat([original, aug_df], ignore_index=True), generated

    return original, 0
