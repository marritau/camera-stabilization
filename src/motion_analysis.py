import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=220,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def generate_demo_video(path: Path, frame_count: int = 140, width: int = 640, height: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (width, height))

    rng = np.random.default_rng(42)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, 24):
        color = 35 + (y // 24) % 2 * 14
        cv2.line(background, (0, y), (width, y), (color, color, color + 10), 1)
    for _ in range(180):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        cv2.circle(background, (x, y), int(rng.integers(1, 3)), (65, 70, 78), -1)

    for frame_idx in range(frame_count):
        frame = background.copy()
        camera_dx = int(10 * np.sin(frame_idx / 18))
        camera_dy = int(5 * np.cos(frame_idx / 24))
        transform = np.float32([[1, 0, camera_dx], [0, 1, camera_dy]])
        frame = cv2.warpAffine(frame, transform, (width, height), borderMode=cv2.BORDER_REFLECT)

        car_x = 30 + int(frame_idx * 3.1)
        car_y = 245 + int(12 * np.sin(frame_idx / 13))
        cv2.rectangle(frame, (car_x, car_y), (car_x + 78, car_y + 34), (40, 165, 230), -1)
        cv2.circle(frame, (car_x + 18, car_y + 36), 9, (20, 20, 22), -1)
        cv2.circle(frame, (car_x + 60, car_y + 36), 9, (20, 20, 22), -1)

        ball_x = 560 - int(frame_idx * 2.2)
        ball_y = 80 + int(45 * np.sin(frame_idx / 11))
        cv2.circle(frame, (ball_x, ball_y), 23, (235, 78, 98), -1)

        person_x = 310 + int(35 * np.sin(frame_idx / 10))
        person_y = 160 + int(frame_idx * 0.55)
        cv2.circle(frame, (person_x, person_y), 16, (90, 220, 130), -1)
        cv2.line(frame, (person_x, person_y + 18), (person_x, person_y + 58), (90, 220, 130), 8)
        cv2.line(frame, (person_x - 26, person_y + 35), (person_x + 26, person_y + 35), (90, 220, 130), 6)

        shadow = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(shadow, (car_x + 42, car_y + 46), (55, 12), 0, 0, 360, 90, -1)
        frame[shadow > 0] = (frame[shadow > 0] * 0.68).astype(np.uint8)

        if 58 <= frame_idx <= 76:
            frame = cv2.GaussianBlur(frame, (7, 7), 0)

        noise = rng.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()


def read_video_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if len(frames) < 2:
        raise ValueError("Video must contain at least two readable frames.")
    return frames, fps


def track_lk(frames: list[np.ndarray]) -> tuple[dict[int, list[tuple[float, float]]], dict[str, float]]:
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(first_gray, mask=None, **FEATURE_PARAMS)
    if points is None:
        return {}, {"initial_points": 0, "final_points": 0, "lost_percent": 100.0, "mean_track_length": 0.0}

    previous_gray = first_gray
    previous_points = points
    trajectories = {idx: [tuple(point.ravel())] for idx, point in enumerate(points)}
    active_ids = np.arange(len(points))

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, **LK_PARAMS)
        if next_points is None or status is None:
            break

        status = status.reshape(-1).astype(bool)
        good_next = next_points[status]
        good_ids = active_ids[status]

        for track_id, point in zip(good_ids, good_next):
            trajectories[int(track_id)].append(tuple(point.ravel()))

        previous_gray = gray
        previous_points = good_next.reshape(-1, 1, 2)
        active_ids = good_ids
        if len(previous_points) == 0:
            break

    lengths = [len(track) for track in trajectories.values()]
    metrics = {
        "initial_points": int(len(points)),
        "final_points": int(len(active_ids)),
        "lost_percent": round(100.0 * (1 - len(active_ids) / len(points)), 2),
        "mean_track_length": round(float(np.mean(lengths)), 2),
        "median_track_length": round(float(np.median(lengths)), 2),
    }
    return trajectories, metrics


def draw_lk_trajectories(frame: np.ndarray, trajectories: dict[int, list[tuple[float, float]]], output_path: Path) -> None:
    canvas = frame.copy()
    rng = np.random.default_rng(7)
    for track in trajectories.values():
        if len(track) < 5:
            continue
        color = tuple(int(value) for value in rng.integers(80, 255, size=3))
        points = np.array(track, dtype=np.int32)
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), 3, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def plot_lk_coordinate_system(
    frame_shape: tuple[int, int, int],
    trajectories: dict[int, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    height, width = frame_shape[:2]
    plt.figure(figsize=(9, 5))
    for track in trajectories.values():
        if len(track) < 12:
            continue
        points = np.array(track)
        plt.plot(points[:, 0], points[:, 1], linewidth=0.8, alpha=0.65)
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.xlabel("x, px")
    plt.ylabel("y, px")
    plt.title("LK trajectories in frame coordinates")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def farneback_analysis(frames: list[np.ndarray], output_dir: Path) -> dict[str, float]:
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    magnitudes = []
    foreground_ratios = []
    component_counts = []
    sample_index = min(35, len(frames) - 2)
    sample_flow = None
    sample_mask = None
    sample_magnitude = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for idx in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_frames[idx], gray_frames[idx + 1], None, **FARNEBACK_PARAMS)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        threshold = max(1.0, float(np.percentile(magnitude, 88)))
        mask = (magnitude > threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        _, labels = cv2.connectedComponents(mask)
        magnitudes.append(float(np.mean(magnitude)))
        foreground_ratios.append(float(np.mean(mask > 0)))
        component_counts.append(int(labels.max()))

        if idx == sample_index:
            sample_flow = flow
            sample_mask = mask
            sample_magnitude = magnitude

    if sample_flow is not None and sample_mask is not None:
        save_flow_hsv(sample_flow, output_dir / "farneback_flow_hsv.png")
        cv2.imwrite(str(output_dir / "farneback_motion_mask.png"), sample_mask)
        save_magnitude_heatmap(sample_magnitude, output_dir / "farneback_magnitude_heatmap.png")

    plt.figure(figsize=(8, 4))
    plt.hist(magnitudes, bins=24, color="#2f6fbb", edgecolor="white")
    plt.xlabel("Mean Farneback magnitude per frame pair, px")
    plt.ylabel("Frame pairs")
    plt.title("Distribution of dense-flow magnitude")
    plt.tight_layout()
    plt.savefig(output_dir / "farneback_magnitude_histogram.png", dpi=160)
    plt.close()

    return {
        "mean_flow_magnitude": round(float(np.mean(magnitudes)), 3),
        "max_mean_flow_magnitude": round(float(np.max(magnitudes)), 3),
        "mean_motion_mask_ratio": round(float(np.mean(foreground_ratios)), 4),
        "mean_motion_components": round(float(np.mean(component_counts)), 2),
    }


def save_flow_hsv(flow: np.ndarray, output_path: Path) -> None:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(output_path), bgr)


def save_magnitude_heatmap(magnitude: np.ndarray, output_path: Path) -> None:
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_path), heatmap)


def write_report(output_dir: Path, video_path: Path, frame_count: int, lk_metrics: dict, farneback_metrics: dict) -> None:
    report = {
        "variant": "B",
        "video": str(video_path),
        "frames_processed": frame_count,
        "lucas_kanade": lk_metrics,
        "farneback": farneback_metrics,
        "interpretation": {
            "lk_strength": "Tracks stable textured corners and gives clean sparse trajectories.",
            "lk_weakness": "Loses points on blur, low texture, object exits, and strong deformation.",
            "farneback_strength": "Produces dense motion everywhere and helps build motion masks.",
            "farneback_weakness": "Adds noisy flow on shadows, global camera drift, blur, and flat background.",
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    markdown = f"""# Motion Analysis Report

Variant: B, educational motion-analysis module.

Video: `{video_path}`
Frames processed: {frame_count}

## Lucas-Kanade

- Initial points: {lk_metrics["initial_points"]}
- Final active points: {lk_metrics["final_points"]}
- Lost points: {lk_metrics["lost_percent"]}%
- Mean track length: {lk_metrics["mean_track_length"]} frames

## Farneback

- Mean dense-flow magnitude: {farneback_metrics["mean_flow_magnitude"]} px
- Max mean dense-flow magnitude: {farneback_metrics["max_mean_flow_magnitude"]} px
- Mean motion-mask area: {farneback_metrics["mean_motion_mask_ratio"] * 100:.2f}%
- Mean connected motion components: {farneback_metrics["mean_motion_components"]}

## Error Analysis

LK works best on corners and textured local structures. It loses tracks when an object leaves the frame,
when motion blur appears, and when local texture is weak. Farneback is better when a dense motion field
or segmentation mask is needed, but it is more sensitive to shadows, camera drift, blur, and background
texture because it estimates a vector for every pixel.
"""
    (output_dir / "analysis_report.md").write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 1, variant B: LK and Farneback motion analysis.")
    parser.add_argument("--input", type=Path, default=None, help="Path to an input video.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for artifacts.")
    parser.add_argument("--demo-video", type=Path, default=Path("data/demo_motion.mp4"), help="Generated demo video path.")
    parser.add_argument("--max-frames", type=int, default=140, help="Number of frames to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.input or args.demo_video
    if args.input is None or not video_path.exists():
        generate_demo_video(video_path, frame_count=args.max_frames)

    frames, _ = read_video_frames(video_path, max_frames=args.max_frames)
    trajectories, lk_metrics = track_lk(frames)
    draw_lk_trajectories(frames[-1], trajectories, args.output_dir / "lk_trajectories_overlay.png")
    plot_lk_coordinate_system(frames[0].shape, trajectories, args.output_dir / "lk_trajectories_coordinates.png")
    farneback_metrics = farneback_analysis(frames, args.output_dir)
    write_report(args.output_dir, video_path, len(frames), lk_metrics, farneback_metrics)

    print(f"Processed {len(frames)} frames from {video_path}")
    print(f"Artifacts saved to {args.output_dir}")
    print(json.dumps({"lucas_kanade": lk_metrics, "farneback": farneback_metrics}, indent=2))


if __name__ == "__main__":
    main()
