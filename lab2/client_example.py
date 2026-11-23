"""Example client: send CIFAR-10 test images to the running service and print the predictions.

    python client_example.py                                  # BentoML on :3111
    python client_example.py --url http://127.0.0.1:8111      # Ray Serve
    python client_example.py --write-curl-payload             # (re)create sample_request.json for cURL

An image travels as base64 of its 3072 raw uint8 bytes (32x32x3, RGB, row-major).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests

from common import MODEL_DIR, encode_image

LAB = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3111")
    parser.add_argument("-n", type=int, default=8, help="number of test images to send")
    parser.add_argument("--write-curl-payload", action="store_true")
    args = parser.parse_args()

    images = np.load(MODEL_DIR / "sample_images.npy")[: args.n]
    labels = np.load(MODEL_DIR / "sample_labels.npy")[: args.n]
    classes = json.loads((MODEL_DIR / "model_info.json").read_text())["classes"]

    if args.write_curl_payload:
        (LAB / "sample_request.json").write_text(json.dumps({"images": [encode_image(images[0])]}))
        print(f"wrote sample_request.json (true label: {classes[labels[0]]})")
        return

    response = requests.post(f"{args.url}/predict", json={"images": [encode_image(img) for img in images]}, timeout=30)
    response.raise_for_status()
    for prediction, label in zip(response.json(), labels):
        mark = "ok " if prediction["class_id"] == label else "BAD"
        print(f"{mark} predicted {prediction['label']:<10} p={prediction['confidence']:.3f}   true: {classes[label]}")


if __name__ == "__main__":
    main()
