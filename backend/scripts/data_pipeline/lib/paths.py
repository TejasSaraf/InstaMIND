

"""
All filesystem paths in one place.
Scripts import from here — never hardcode paths in pipeline scripts.
"""

from pathlib import Path







PROJECT_ROOT = Path(__file__).resolve().parents[4]





UCF_ROOT = PROJECT_ROOT / "data" / "raw" / "ucf_crimes"

UCA_ROOT = PROJECT_ROOT / "data" / "raw" / "uca_annotations"















URFD_ROOT = UCF_ROOT / "Fall"





MANIFESTS = PROJECT_ROOT / "data" / "processed" / "manifests"

FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "frames"

AUGMENTED = PROJECT_ROOT / "data" / "processed" / "augmented"





ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"





for _d in [MANIFESTS, FRAMES_ROOT, AUGMENTED, ANNOTATIONS]:

    _d.mkdir(parents=True, exist_ok=True)