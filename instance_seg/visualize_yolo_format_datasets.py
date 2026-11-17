import argparse
from pathlib import Path

import cv2
import numpy as np


CHANNEL_NAMES = ["VD", "CHM", "G_CHM", "LBP_CHM"]


def load_feature(feature_path):
    feature = np.load(feature_path)
    if feature.ndim != 3:
        raise ValueError(f"{feature_path} should be 3D array")
    if feature.shape[0] == 4:
        return feature
    if feature.shape[-1] == 4:
        return np.transpose(feature, (2, 0, 1))
    raise ValueError(f"{feature_path} should have 4 channels")


def normalize_channel(channel, percentile_low, percentile_high):
    channel = channel.astype(np.float32)
    low = np.percentile(channel, percentile_low)
    high = np.percentile(channel, percentile_high)
    if high <= low:
        return np.zeros_like(channel, dtype=np.uint8)
    channel = np.clip(channel, low, high)
    channel = (channel - low) / (high - low)
    return (channel * 255).clip(0, 255).astype(np.uint8)


def make_rgb(feature, channel_indices, percentile_low, percentile_high):
    rgb_channels = []
    for channel_index in channel_indices:
        rgb_channels.append(normalize_channel(feature[channel_index], percentile_low, percentile_high))
    rgb = np.stack(rgb_channels, axis=-1)
    return rgb


def read_yolo_labels(label_path):
    if not label_path.exists():
        return []

    labels = []
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        labels.append((cls_id, x_center, y_center, width, height))
    return labels


def draw_boxes(image, labels, color, thickness, show_id):
    image_h, image_w = image.shape[:2]
    output = image.copy()

    for box_index, label in enumerate(labels):
        cls_id, x_center, y_center, width, height = label

        x1 = int(round((x_center - width / 2.0) * image_w))
        y1 = int(round((y_center - height / 2.0) * image_h))
        x2 = int(round((x_center + width / 2.0) * image_w))
        y2 = int(round((y_center + height / 2.0) * image_h))

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w - 1))
        y2 = max(0, min(y2, image_h - 1))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        if show_id:
            text = f"{cls_id}:{box_index}"
            cv2.putText(output, text, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return output


def draw_centers(image, labels, color, radius, show_id):
    image_h, image_w = image.shape[:2]
    output = image.copy()

    for box_index, label in enumerate(labels):
        cls_id, x_center, y_center, width, height = label

        cx = int(round(x_center * image_w))
        cy = int(round(y_center * image_h))
        cx = max(0, min(cx, image_w - 1))
        cy = max(0, min(cy, image_h - 1))

        cv2.circle(output, (cx, cy), radius, color, -1, cv2.LINE_AA)

        if show_id:
            text = f"C{box_index}"
            cv2.putText(output, text, (min(image_w - 1, cx + radius + 2), min(image_h - 1, cy + radius + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return output


def parse_channels(channels):
    values = [item.strip() for item in channels.split(",")]
    if len(values) != 3:
        raise ValueError("--channels should contain exactly 3 values, for example 0,1,2 or VD,CHM,LBP_CHM")

    indices = []
    for value in values:
        if value.isdigit():
            index = int(value)
        else:
            if value not in CHANNEL_NAMES:
                raise ValueError(f"Unknown channel name: {value}")
            index = CHANNEL_NAMES.index(value)
        if index < 0 or index >= 4:
            raise ValueError(f"Channel index out of range: {index}")
        indices.append(index)

    return indices


def collect_feature_files(image_dir, limit):
    files = sorted(image_dir.glob("*.npy"))
    if limit > 0:
        files = files[:limit]
    return files


def parse_color(color):
    values = [int(value.strip()) for value in color.split(",")]
    if len(values) != 3:
        raise ValueError("--center-color should contain exactly 3 values, for example 0,0,0")
    return tuple(max(0, min(value, 255)) for value in values)


def main():
    parser = argparse.ArgumentParser(description="可视化 YOLO 格式的 4 通道橄榄数据集标签框和框中心质心点")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/yolo_olive_4ch"), help="YOLO 格式数据集根目录，目录下应包含 images/{split} 和 labels/{split}")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="需要可视化的数据划分，可选 train、val 或 test")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/yolo_olive_4ch/visual_check"), help="可视化图片输出根目录，脚本会在该目录下按 split 新建子目录")
    parser.add_argument("--channels", type=str, default="VD,CHM,LBP_CHM", help="用于合成 RGB 的 3 个通道名称或索引，逗号分隔，可选 VD、CHM、G_CHM、LBP_CHM 或 0、1、2、3")
    parser.add_argument("--percentile-low", type=float, default=1.0, help="通道归一化的低百分位截断值，用于增强显示对比度")
    parser.add_argument("--percentile-high", type=float, default=99.0, help="通道归一化的高百分位截断值，用于增强显示对比度")
    parser.add_argument("--limit", type=int, default=0, help="最多可视化多少个文件，0 表示不限制数量")
    parser.add_argument("--box-thickness", type=int, default=2, help="绘制 YOLO 标签框的线宽，单位为像素")
    parser.add_argument("--show-id", action="store_true", help="是否在标签框和中心点旁显示类别与序号")
    parser.add_argument("--draw-centers", action="store_true", help="是否在每个 YOLO 标签框中心绘制质心点")
    parser.add_argument("--center-radius", type=int, default=5, help="质心点半径，单位为像素")
    parser.add_argument("--center-color", type=str, default="0,0,0", help="质心点颜色，格式为 R,G,B，例如 0,0,0 表示黑色")
    args = parser.parse_args()

    image_dir = args.dataset_dir / "images" / args.split
    label_dir = args.dataset_dir / "labels" / args.split
    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_indices = parse_channels(args.channels)
    center_color = parse_color(args.center_color)
    feature_files = collect_feature_files(image_dir, args.limit)

    if not feature_files:
        raise FileNotFoundError(f"No .npy files found in {image_dir}")

    for feature_path in feature_files:
        feature = load_feature(feature_path)
        rgb = make_rgb(feature, channel_indices, args.percentile_low, args.percentile_high)

        label_path = label_dir / f"{feature_path.stem}.txt"
        labels = read_yolo_labels(label_path)

        visual = draw_boxes(
            image=rgb,
            labels=labels,
            color=(255, 0, 0),
            thickness=args.box_thickness,
            show_id=args.show_id,
        )

        if args.draw_centers:
            visual = draw_centers(
                image=visual,
                labels=labels,
                color=center_color,
                radius=args.center_radius,
                show_id=args.show_id,
            )

        output_path = output_dir / f"{feature_path.stem}_{args.channels.replace(',', '-')}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
        print(f"[OK] {feature_path.name}: boxes={len(labels)} -> {output_path}")


if __name__ == "__main__":
    main()