"""
models/downloader.py

Utility to download MediaPipe task model files into the project's `models/` folder.

This tries a set of candidate URLs for each required model and saves them locally.
If downloads fail, it prints instructions so you can download models manually.
"""
import os
import urllib.request

MODELS = {
    "blaze_face_short_range.tflite": [
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ],
    "face_landmarker_v2_with_blendshapes.task": [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/1/face_landmarker_v2_with_blendshapes.task",
    ],
}


def ensure_models(dest_dir:
                   str = "models") -> dict:
    """Ensure model files exist in `dest_dir`. Returns dict of model_name -> path or None.

    The function will attempt to download missing models from candidate URLs.
    It will not overwrite existing files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    results = {}

    for model_name, urls in MODELS.items():
        dest_path = os.path.join(dest_dir, model_name)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            results[model_name] = dest_path
            continue

        success = False
        for url in urls:
            try:
                print(f"Attempting download: {url} -> {dest_path}")
                # try a simple HTTP GET and follow redirects
                urllib.request.urlretrieve(url, dest_path)
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    print(f"Downloaded {model_name} from {url}")
                    results[model_name] = dest_path
                    success = True
                    break
            except Exception as e:
                print(f"Failed to download {url}: {e}")

        if not success:
            print(f"Could not download {model_name}. Please obtain it and place it at: {dest_path}")
            print("Common places to check:")
            print(" - https://github.com/google/mediapipe-models/releases")
            print(" - https://storage.googleapis.com/mediapipe-assets/")
            print(f"Save the model file named '{model_name}' into the '{dest_dir}' folder.")
            results[model_name] = None

    return results


if __name__ == "__main__":
    ensure_models()
