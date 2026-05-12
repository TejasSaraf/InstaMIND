"""Ensure uploaded media is decodable by OpenCV/librosa before analysis."""



from __future__ import annotations



import os

import shutil

import subprocess

import tempfile

from pathlib import Path



import cv2





def resolve_media_path(path: Path) -> Path:

    return path.expanduser().resolve()





def _cv2_can_decode(path: Path) -> bool:

    abspath = str(resolve_media_path(path))

    cap = cv2.VideoCapture(abspath, cv2.CAP_FFMPEG)

    if not cap.isOpened():

        return False

    try:

        ok, frame = cap.read()

        return bool(ok and frame is not None)

    finally:

        cap.release()





def _run_ffmpeg(

    src: Path,

    dst: Path,

    *,

    video_only: bool,

    force_format: str | None = None,

) -> None:

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:

        raise FileNotFoundError("ffmpeg")

    src_abs = str(resolve_media_path(src))

    cmd: list[str] = [

        ffmpeg,

        "-y",

        "-hide_banner",

        "-loglevel",

        "error",

        "-probesize",

        "50M",

        "-analyzeduration",

        "100M",

    ]

    if force_format:

        cmd.extend(["-f", force_format])

    cmd.extend(

        [

            "-fflags",

            "+genpts+discardcorrupt",

            "-err_detect",

            "ignore_err",

            "-i",

            src_abs,

            "-c:v",

            "libx264",

            "-preset",

            "ultrafast",

            "-pix_fmt",

            "yuv420p",

            "-movflags",

            "+faststart",

        ]

    )

    if video_only:

        cmd.extend(["-map", "0:v:0", "-an"])

    else:

        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.append(str(dst))

    subprocess.run(cmd, check=True, capture_output=True, timeout=300)





def _try_transcode(src: Path) -> Path | None:

    if not shutil.which("ffmpeg"):

        return None

    src = resolve_media_path(src)

    try:

        if not src.is_file() or src.stat().st_size < 32:

            return None

    except OSError:

        return None



    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="instamind_norm_")

    os.close(fd)

    out = Path(tmp)



    suffix = src.suffix.lower()

    fmt_hints: list[str | None] = [None]

    if suffix in {".webm", ".mkv"}:

        fmt_hints.extend(["matroska", "webm"])



    for video_only in (False, True):

        for fmt in fmt_hints:

            try:

                _run_ffmpeg(src, out, video_only=video_only, force_format=fmt)

                if out.exists() and out.stat().st_size > 256:

                    return out

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, FileNotFoundError):

                pass

            out.unlink(missing_ok=True)

    return None





def transcode_video_to_temp_mp4(src: Path) -> Path | None:

    """
    FFmpeg transcode to a temp MP4. Use when OpenCV seek/read is unreliable
    (browser WebM/Matroska segments).
    """

    return _try_transcode(resolve_media_path(src))





def ensure_decodable_video(path: Path) -> tuple[Path, Path | None]:

    """
    Return (path_to_analyze, temp_path_or_none).
    If transcoding is used, caller must delete temp_path after analysis.
    """

    path = resolve_media_path(path)

    if _cv2_can_decode(path):

        return path, None

    converted = _try_transcode(path)

    if converted is None:

        raise ValueError(f"Unable to open video: {path}")

    return converted, converted
