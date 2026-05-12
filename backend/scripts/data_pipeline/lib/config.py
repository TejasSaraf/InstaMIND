

"""
Externalized pipeline configuration.

All tunable parameters in one place. Override via CLI or config file
without touching pipeline code.

Usage:
    cfg = PipelineConfig.from_cli()       # with --cap-percentile 80 etc.
    cfg = PipelineConfig()                # defaults
    cfg = PipelineConfig.from_dict({...}) # from JSON/dict
"""



from __future__ import annotations

import argparse

import hashlib

import json

from dataclasses import dataclass, field, asdict

from pathlib import Path





@dataclass(frozen=True)

class PipelineConfig:

    """All pipeline hyperparameters. Frozen — cannot be mutated after creation."""





    dhash_size: int              = 16

    hamming_thresh_train: int    = 20

    hamming_thresh_valtest: int  = 0

    ssim_thresh: float           = 0.85

    sliding_window_size: int     = 10

    min_temporal_gap_s: float    = 0.5





    cap_percentile: int          = 75

    cap_max_ratio: float         = 0.5

    min_floor: int               = 400

    random_seed: int             = 42





    jpeg_quality: int            = 92

    aug_brightness_range: float  = 0.2

    aug_contrast_range: float    = 0.2

    aug_rotation_deg: float      = 5.0

    aug_flip_prob: float         = 0.5

    aug_blur_prob: float         = 0.3

    aug_blur_radius: float       = 1.0





    min_samples_per_class: int   = 50

    adaptive_min_enabled: bool   = True

    strict_identity_check: bool  = False





    critical_classes: tuple       = ("shooting", "fainting")

    critical_class_min_floor: int = 500

    critical_class_weight_boost: float = 2.0





    log_json: bool               = True



    def to_dict(self) -> dict:

        return asdict(self)



    def fingerprint(self) -> str:

        """8-char hex digest of config — used to version augmented data."""

        raw = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")

        return hashlib.md5(raw).hexdigest()[:8]



    def save(self, path: Path) -> None:

        path.write_text(json.dumps(self.to_dict(), indent=2))



    @classmethod

    def from_dict(cls, d: dict) -> PipelineConfig:



        valid = {f.name for f in cls.__dataclass_fields__.values()}

        filtered = {k: v for k, v in d.items() if k in valid}



        if "critical_classes" in filtered and isinstance(filtered["critical_classes"], list):

            filtered["critical_classes"] = tuple(filtered["critical_classes"])

        return cls(**filtered)



    @classmethod

    def from_json(cls, path: Path) -> PipelineConfig:

        return cls.from_dict(json.loads(path.read_text()))



    @classmethod

    def from_cli(cls) -> PipelineConfig:

        """Parse CLI args. Any unset arg uses the dataclass default."""

        parser = argparse.ArgumentParser(

            description="Phase 5: Verify → Deduplicate → Balance"

        )





        parser.add_argument("--hamming-thresh-train", type=int, default=None)

        parser.add_argument("--hamming-thresh-valtest", type=int, default=None)

        parser.add_argument("--ssim-thresh", type=float, default=None)

        parser.add_argument("--sliding-window-size", type=int, default=None)

        parser.add_argument("--min-temporal-gap", type=float, default=None)





        parser.add_argument("--cap-percentile", type=int, default=None)

        parser.add_argument("--cap-max-ratio", type=float, default=None)

        parser.add_argument("--min-floor", type=int, default=None)

        parser.add_argument("--seed", type=int, default=None)





        parser.add_argument("--critical-class-min-floor", type=int, default=None,

                            help="Minimum samples for safety-critical classes")

        parser.add_argument("--critical-class-weight-boost", type=float, default=None)





        parser.add_argument("--strict-identity-check", action="store_true")





        parser.add_argument("--config", type=str, default=None,

                            help="Path to JSON config file (overrides all defaults)")

        parser.add_argument("--no-log-json", action="store_true")



        args = parser.parse_args()





        if args.config:

            base = cls.from_json(Path(args.config)).to_dict()

        else:

            base = cls().to_dict()





        cli_map = {

            "hamming_thresh_train": args.hamming_thresh_train,

            "hamming_thresh_valtest": args.hamming_thresh_valtest,

            "ssim_thresh": args.ssim_thresh,

            "sliding_window_size": args.sliding_window_size,

            "min_temporal_gap_s": args.min_temporal_gap,

            "cap_percentile": args.cap_percentile,

            "cap_max_ratio": args.cap_max_ratio,

            "min_floor": args.min_floor,

            "random_seed": args.seed,

        }



        for k, v in cli_map.items():

            if v is not None:

                base[k] = v



        if args.critical_class_min_floor is not None:

            base["critical_class_min_floor"] = args.critical_class_min_floor

        if args.critical_class_weight_boost is not None:

            base["critical_class_weight_boost"] = args.critical_class_weight_boost

        if args.strict_identity_check:

            base["strict_identity_check"] = True

        if args.no_log_json:

            base["log_json"] = False





        return cls.from_dict(base)
