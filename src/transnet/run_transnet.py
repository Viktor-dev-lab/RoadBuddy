import sys
import os
import cv2

# ===== Resolve paths safely =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

TRANSNET_INFERENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "third_party",
    "TransNetV2",
    "inference"
)

sys.path.insert(0, TRANSNET_INFERENCE_DIR)

from transnetv2 import TransNetV2

MODEL_DIR = os.path.join(
    TRANSNET_INFERENCE_DIR,
    "transnetv2-weights"
)

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/videos/Dashcam2.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/images")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_scene_frames_fullres(video_path, scenes):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    for scene_id, (start, end) in enumerate(scenes):
        if end <= start:
            continue

        mid = (start + end) // 2
        frame_indices = {
            "start": start,
            "mid": mid,
            "end": end - 1
        }

        for pos, idx in frame_indices.items():
            if idx < 0 or idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            filename = f"scene_{scene_id:04d}_{pos}.jpg"
            out_path = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(out_path, frame)

    cap.release()
    print(f"Saved FULL-RES scene images to: {OUTPUT_DIR}")



def main():
    print("Loading TransNetV2 model...")
    model = TransNetV2(model_dir=MODEL_DIR)
    print("Model loaded!")

    print("Processing video:", VIDEO_PATH)
    video_frames, single_preds, all_preds = model.predict_video(VIDEO_PATH)

    scenes = model.predictions_to_scenes(single_preds)
    print(f"Detected {len(scenes)} scenes")

    save_scene_frames_fullres(VIDEO_PATH, scenes)


if __name__ == "__main__":
    main()
