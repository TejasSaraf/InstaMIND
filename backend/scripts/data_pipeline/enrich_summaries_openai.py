from lib.paths import MANIFESTS, ANNOTATIONS
import os

import sys

import json

import base64

import pandas as pd

from pathlib import Path

from tqdm import tqdm

from openai import OpenAI


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parent))


INPUT_PARQUET = MANIFESTS / "frames_final.parquet"

OUTPUT_JSON = ANNOTATIONS / "openai_summaries.json"


MAX_FRAMES_PER_CALL = 6


def encode_image_to_base64(image_path: str) -> str:

    with open(image_path, "rb") as image_file:

        return base64.b64encode(image_file.read()).decode('utf-8')


def load_env_file():
    """Load variables from backend/.env into os.environ"""

    env_path = Path(__file__).resolve().parents[2] / ".env"

    if env_path.exists():

        with open(env_path, "r") as f:

            for line in f:

                line = line.strip()

                if line and not line.startswith("#"):

                    if "=" in line:

                        k, v = line.split("=", 1)

                        v = v.strip('"\'')

                        os.environ[k.strip()] = v


def main():

    load_env_file()

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:

        print("Error: OPENAI_API_KEY environment variable is not set.")

        print("Please add OPENAI_API_KEY='your-key-here' to backend/.env")

        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading {INPUT_PARQUET}...")

    df = pd.read_parquet(INPUT_PARQUET)

    df = df[~df["frame_path"].str.contains("_aug_")]

    groups = df.groupby(["video_id", "window_start_s"])

    existing_summaries = {}

    if OUTPUT_JSON.exists():

        with open(OUTPUT_JSON, "r") as f:

            existing_summaries = json.load(f)

        print(f"Loaded {len(existing_summaries)} existing summaries.")

    results = existing_summaries.copy()

    tasks = []

    for (vid, w_start), group in groups:

        key = f"{vid}_{w_start}"

        if key in results:

            continue

        tasks.append((key, vid, w_start, group))

    print(f"Generating summaries for {len(tasks)} windows...")

    for key, vid, w_start, group in tqdm(tasks, desc="OpenAI Generation"):

        incident_type = group["incident_type"].iloc[0]

        original_desc = group["description"].iloc[0]

        group = group.sort_values("frame_idx")

        frame_paths = group["frame_path"].tolist()

        if len(frame_paths) > MAX_FRAMES_PER_CALL:

            indices = [int(i * (len(frame_paths) - 1) / (MAX_FRAMES_PER_CALL - 1))
                       for i in range(MAX_FRAMES_PER_CALL)]

            selected_paths = [frame_paths[i] for i in indices]

        else:

            selected_paths = frame_paths

        prompt_text = (

            f"You are an expert security surveillance analyst. I am providing you with a temporal sequence of {len(selected_paths)} frames "

            f"from a security camera that captures a '{incident_type}' incident.\n\n"

            f"The original human annotator described this scene as: '{original_desc}'.\n\n"

            "Task: Write a single, highly detailed, objective, and grammatically correct sentence describing the specific actions taking place in these frames. "

            "Focus on the chronological flow of actions, the people involved, and what they are doing. "

            "Do NOT mention the camera, the fact that these are frames/video, and do not make assumptions beyond what is visible. "

            "Return ONLY the sentence."

        )

        content_list = [{"type": "text", "text": prompt_text}]

        valid_paths = [fp for fp in selected_paths if Path(fp).exists()]

        if not valid_paths:

            continue

        for fp in valid_paths:

            b64_img = encode_image_to_base64(fp)

            content_list.append({

                "type": "image_url",

                "image_url": {

                    "url": f"data:image/jpeg;base64,{b64_img}",

                    "detail": "low"

                }

            })

        try:

            response = client.chat.completions.create(

                model="gpt-4o",

                messages=[

                    {"role": "user", "content": content_list}

                ],

                max_tokens=150,

                temperature=0.3,

            )

            new_summary = response.choices[0].message.content.strip()

            if new_summary.startswith('"') and new_summary.endswith('"'):

                new_summary = new_summary[1:-1]

            results[key] = new_summary

            with open(OUTPUT_JSON, "w") as f:

                json.dump(results, f, indent=2)

        except Exception as e:

            print(f"\nError processing {key}: {e}")

            pass

    print(f"\nFinished! Total summaries: {len(results)}")

    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":

    main()
