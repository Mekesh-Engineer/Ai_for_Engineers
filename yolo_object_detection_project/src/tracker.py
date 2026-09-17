# -*- coding: utf-8 -*-
"""
tracker.py
----------
Real-Time Moving Object Detection and Multi-Object Tracking (MOT) on video streams
and webcam feeds using Ultralytics YOLO and ByteTrack / BoT-SORT.
Tracks persistent identities across frames, visualizes motion trajectories,
and exports annotated video recordings.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


def track_moving_objects_video(
    video_source: str,
    output_video_path: str = "results/tracked_output.mp4",
    model_path: str = "models/best.pt",
    tracker: str = "bytetrack.yaml",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: list = None,
    draw_trajectories: bool = True
) -> dict:
    """
    Process video stream, detect moving objects, and track unique IDs across frames.

    Parameters
    ----------
    video_source : str
        Path to input video file or '0' for live webcam.
    output_video_path : str
        Filepath to write rendered tracked MP4 output.
    model_path : str
        Path to trained YOLO model weights.
    tracker : str
        Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
    conf_threshold : float
        Detection confidence cutoff.
    iou_threshold : float
        NMS IoU overlap cutoff.
    target_classes : list
        List of class IDs to track (e.g. [0] for person only).
    draw_trajectories : bool
        Whether to draw historical motion trail paths.

    Returns
    -------
    dict
        Tracking execution statistics (total frames, unique IDs, avg FPS).
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Fallback to yolov8n.pt if model_path does not exist
    if not os.path.exists(model_path):
        print(f"[Tracker] Model '{model_path}' not found. Using pre-trained 'yolov8n.pt'...")
        model_path = "yolov8n.pt"

    print(f"[Tracker] Loading YOLO model from '{model_path}' with '{tracker}'...")
    model = YOLO(model_path)

    # Open video capture
    is_webcam = str(video_source).isdigit()
    cap_src = int(video_source) if is_webcam else video_source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        raise IOError(f"[Tracker] Failed to open video source: '{video_source}'")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Track historical bottom-center coordinates: {track_id: [(x, y), ...]}
    track_history = {}
    unique_ids_encountered = set()
    frame_count = 0
    processing_times = []

    print(f"[Tracker] Processing video stream: {width}x{height} @ {fps:.1f} FPS...")

    try:
        while cap.isOpened():
            t_start = cv2.getTickCount()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Execute tracking
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=target_classes,
                verbose=False
            )[0]

            annotated_frame = frame.copy()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                classes = results.boxes.cls.cpu().numpy().astype(int)
                scores = results.boxes.conf.cpu().numpy()

                for box, track_id, cls_id, score in zip(boxes, track_ids, classes, scores):
                    unique_ids_encountered.add(track_id)
                    x1, y1, x2, y2 = box
                    cname = model.names.get(cls_id, f"cls_{cls_id}")

                    # Generate distinct pseudo-random color for each track ID
                    color = (
                        int((track_id * 63 + 50) % 255),
                        int((track_id * 117 + 80) % 255),
                        int((track_id * 179 + 110) % 255)
                    )

                    # Bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Label badge
                    label_text = f"ID #{track_id} {cname} {score*100.0:.0f}%"
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    badge_y1 = max(0, y1 - th - 6)
                    cv2.rectangle(annotated_frame, (x1, badge_y1), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated_frame, label_text, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # Draw motion trajectory trail
                    if draw_trajectories:
                        center_pt = (int((x1 + x2) / 2), int(y2))  # Bottom-center
                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append(center_pt)
                        if len(track_history[track_id]) > 35:
                            track_history[track_id].pop(0)

                        if len(track_history[track_id]) > 1:
                            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)

            t_end = cv2.getTickCount()
            dt_ms = (t_end - t_start) / cv2.getTickFrequency() * 1000.0
            processing_times.append(dt_ms)
            cur_fps = 1000.0 / dt_ms if dt_ms > 0 else 30.0

            # HUD Display
            hud_text = f"Frame: {frame_count} | Active: {len(results.boxes.id) if results.boxes.id is not None else 0} | Total IDs: {len(unique_ids_encountered)} | Speed: {cur_fps:.1f} FPS"
            cv2.rectangle(annotated_frame, (10, 10), (width - 10, 45), (30, 30, 30), -1)
            cv2.putText(annotated_frame, hud_text, (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

            out.write(annotated_frame)

    finally:
        cap.release()
        out.release()

    avg_latency = float(np.mean(processing_times)) if len(processing_times) > 0 else 10.0
    effective_fps = 1000.0 / avg_latency if avg_latency > 0 else 30.0

    print(f"\n[Tracker] Completed video tracking:")
    print(f"  - Total Processed Frames  : {frame_count}")
    print(f"  - Unique Tracked Persons  : {len(unique_ids_encountered)}")
    print(f"  - Average Processing Speed: {effective_fps:.1f} FPS ({avg_latency:.2f} ms/frame)")
    print(f"  - Saved Tracked Video     : '{output_video_path}'\n")

    return {
        "frames": frame_count,
        "unique_objects": len(unique_ids_encountered),
        "fps": effective_fps,
        "output_path": output_video_path
    }


def create_synthetic_moving_pedestrian_video(output_path: str = "data/sample_moving_persons.mp4", num_frames: int = 75) -> str:
    """
    Generate a test video of moving pedestrians by animating real-world pedestrian crops
    across a street scene for multi-object tracking validation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 640, 480
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Look for real pedestrian images in PennFudan or data/images/test
    ped_patches = []
    real_img_dirs = ["data/raw_penn_fudan/PennFudanPed/PNGImages", "data/images/test", "data/images/val"]
    
    for r_dir in real_img_dirs:
        if os.path.exists(r_dir):
            for fname in sorted(os.listdir(r_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(r_dir, fname)
                    img = cv2.imread(fpath)
                    if img is not None:
                        # Extract a person crop from center/lower area
                        ih, iw = img.shape[:2]
                        crop = img[int(ih*0.1):int(ih*0.9), int(iw*0.2):int(iw*0.8)]
                        if crop.size > 0:
                            ped_patches.append(crop)
                            if len(ped_patches) >= 3:
                                break
        if len(ped_patches) >= 3:
            break

    # If no real images found, prepare fallback synthetic silhouettes
    if len(ped_patches) == 0:
        for _ in range(3):
            dummy_patch = np.zeros((160, 70, 3), dtype=np.uint8)
            dummy_patch[:] = (0, 180, 240)
            cv2.circle(dummy_patch, (35, 25), 20, (0, 210, 255), -1)
            ped_patches.append(dummy_patch)

    # Initialize moving pedestrians
    ped_tracks = [
        {"patch": cv2.resize(ped_patches[0 % len(ped_patches)], (90, 200)), "x": 40.0, "y": 180.0, "vx": 4.5, "vy": 0.3},
        {"patch": cv2.resize(ped_patches[1 % len(ped_patches)], (80, 180)), "x": 520.0, "y": 200.0, "vx": -3.8, "vy": 0.2},
        {"patch": cv2.resize(ped_patches[2 % len(ped_patches)], (100, 220)), "x": 160.0, "y": 220.0, "vx": 2.2, "vy": -0.2}
    ]

    for frame_idx in range(num_frames):
        # Create background street / sidewalk
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:int(h*0.45)] = (180, 160, 140)     # Sky/buildings
        canvas[int(h*0.45):int(h*0.65)] = (130, 130, 130)  # Sidewalk
        canvas[int(h*0.65):] = (70, 70, 70)         # Road asphalt
        cv2.line(canvas, (0, int(h*0.45)), (w, int(h*0.45)), (110, 110, 110), 2)
        cv2.line(canvas, (0, int(h*0.65)), (w, int(h*0.65)), (220, 220, 220), 3)

        for p in ped_tracks:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            px = int(round(p["x"]))
            py = int(round(p["y"]))
            patch = p["patch"]
            ph, pw = patch.shape[:2]

            # Composite onto canvas
            x1, y1 = max(0, px), max(0, py)
            x2, y2 = min(w, px + pw), min(h, py + ph)

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w > 10 and crop_h > 10:
                patch_x1 = 0 if px >= 0 else -px
                patch_y1 = 0 if py >= 0 else -py
                patch_crop = patch[patch_y1:patch_y1 + crop_h, patch_x1:patch_x1 + crop_w]
                
                # Alpha blend patch onto background
                canvas[y1:y2, x1:x2] = patch_crop

        out.write(canvas)

    out.release()
    print(f"[Tracker] Created moving pedestrians test video at '{output_path}'")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Moving Object Tracking Runner")
    parser.add_argument("--source", type=str, default=None, help="Path to video file or '0' for webcam")
    parser.add_argument("--output", type=str, default="results/tracked_output.mp4", help="Output MP4 path")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model weights")
    args = parser.parse_args()

    src = args.source
    if src is None or not os.path.exists(src):
        src = create_synthetic_moving_pedestrian_video()

    track_moving_objects_video(video_source=src, output_video_path=args.output, model_path=args.model)
