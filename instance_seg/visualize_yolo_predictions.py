import argparse
from pathlib import Path

import cv2
import numpy as np


CHANNEL_NAMES = ["VD", "CHM", "G_CHM", "LBP_CHM"]


def load_feature(feature_path):
    feature = np.load(feature_path)
    if feature.ndim != 3:
        raise ValueError(f"{feature_path} should be 3D")
    if feature.shape[0] == 4:
        return feature
    if feature.shape[-1] == 4:
        return np.transpose(feature, (2, 0, 1))
    raise ValueError(f"{feature_path} should have 4 channels, got {feature.shape}")


def parse_channels(channels):
    values = [value.strip() for value in channels.split(",")]
    if len(values) != 3:
        raise ValueError("--channels should contain exactly 3 values")

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


def parse_color(color):
    values = [int(value.strip()) for value in color.split(",")]
    if len(values) != 3:
        raise ValueError("color should contain exactly 3 values, such as 0,0,0")
    return tuple(max(0, min(value, 255)) for value in values)


def normalize_channel(channel, low_percentile, high_percentile):
    channel = channel.astype(np.float32)
    low = np.percentile(channel, low_percentile)
    high = np.percentile(channel, high_percentile)

    if high <= low:
        return np.zeros_like(channel, dtype=np.uint8)

    channel = np.clip(channel, low, high)
    channel = (channel - low) / (high - low)
    return (channel * 255).clip(0, 255).astype(np.uint8)


def make_rgb(feature, channel_indices, low_percentile, high_percentile):
    channels = [
        normalize_channel(feature[index], low_percentile, high_percentile)
        for index in channel_indices
    ]
    return np.stack(channels, axis=-1)


def read_yolo_labels(label_path, has_conf, conf_threshold):
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
        confidence = float(parts[5]) if has_conf and len(parts) >= 6 else None

        if confidence is not None and confidence < conf_threshold:
            continue

        labels.append((cls_id, x_center, y_center, width, height, confidence))

    return labels


def draw_boxes(image, labels, color, thickness, prefix, show_conf):
    output = image.copy()
    image_h, image_w = output.shape[:2]

    for index, label in enumerate(labels):
        cls_id, x_center, y_center, width, height, confidence = label

        x1 = int(round((x_center - width / 2) * image_w))
        y1 = int(round((y_center - height / 2) * image_h))
        x2 = int(round((x_center + width / 2) * image_w))
        y2 = int(round((y_center + height / 2) * image_h))

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w - 1))
        y2 = max(0, min(y2, image_h - 1))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        text = f"{prefix}{index}"
        if show_conf and confidence is not None:
            text = f"{prefix}{index}:{confidence:.2f}"

        cv2.putText(
            output,
            text,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def draw_centers(image, labels, color, radius, prefix, show_conf):
    output = image.copy()
    image_h, image_w = output.shape[:2]

    for index, label in enumerate(labels):
        cls_id, x_center, y_center, width, height, confidence = label

        cx = int(round(x_center * image_w))
        cy = int(round(y_center * image_h))

        cx = max(0, min(cx, image_w - 1))
        cy = max(0, min(cy, image_h - 1))

        cv2.circle(output, (cx, cy), radius, color, -1, cv2.LINE_AA)

        text = f"{prefix}{index}"
        if show_conf and confidence is not None:
            text = f"{prefix}{index}:{confidence:.2f}"

        cv2.putText(
            output,
            text,
            (min(image_w - 1, cx + radius + 2), min(image_h - 1, cy + radius + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def collect_feature_files(image_dir, pattern, limit):
    files = sorted(image_dir.glob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def main():
    parser = argparse.ArgumentParser(description="将 4 通道橄榄特征图转换为 RGB 可视化图，并叠加 YOLO 预测框、真实框和预测中心点")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/yolo_olive_4ch"), help="YOLO 格式数据集根目录，目录下应包含 images/{split} 和 labels/{split}")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="需要可视化的数据划分，可选 train、val 或 test")
    parser.add_argument("--pred-label-dir", type=Path, default=Path("/workspace/Yolo_liqiying/runs_olive_4ch_predict/best_test_pred/labels"), help="YOLO 预测结果 txt 标签目录，通常是 predict 输出目录下的 labels 文件夹")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/runs_olive_4ch_predict/visual_best_test_pred"), help="可视化图片输出根目录，脚本会在该目录下按 split 新建子目录")
    parser.add_argument("--channels", type=str, default="CHM,G_CHM,LBP_CHM", help="用于合成 RGB 的 3 个通道名称或索引，逗号分隔，可选 VD、CHM、G_CHM、LBP_CHM 或 0、1、2、3")
    parser.add_argument("--pattern", type=str, default="*.npy", help="需要读取的特征文件匹配模式，默认读取所有 .npy 文件")
    parser.add_argument("--limit", type=int, default=0, help="最多可视化多少个文件，0 表示不限制数量")
    parser.add_argument("--conf-threshold", type=float, default=0.10, help="预测框置信度阈值，低于该值的预测框不会被绘制")
    parser.add_argument("--percentile-low", type=float, default=1.0, help="通道归一化的低百分位截断值，用于增强显示对比度")
    parser.add_argument("--percentile-high", type=float, default=99.0, help="通道归一化的高百分位截断值，用于增强显示对比度")
    parser.add_argument("--box-thickness", type=int, default=2, help="绘制真实框和预测框时的线宽，单位为像素")
    parser.add_argument("--draw-gt", action="store_true", help="是否绘制数据集 labels/{split} 中的真实框，绿色显示")
    parser.add_argument("--draw-pred", action="store_true", help="是否绘制 pred-label-dir 中的预测框，红色显示")
    parser.add_argument("--show-conf", action="store_true", help="是否在预测框或中心点标签上显示置信度")
    parser.add_argument("--draw-centers", action="store_true", help="是否在每个预测框中心绘制中心点")
    parser.add_argument("--center-radius", type=int, default=6, help="预测框中心点半径，单位为像素")
    parser.add_argument("--center-color", type=str, default="0,0,0", help="预测框中心点颜色，格式为 R,G,B，例如 0,0,0 表示黑色")
    args = parser.parse_args()

    image_dir = args.dataset_dir / "images" / args.split
    gt_label_dir = args.dataset_dir / "labels" / args.split
    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_indices = parse_channels(args.channels)
    center_color = parse_color(args.center_color)
    feature_files = collect_feature_files(image_dir, args.pattern, args.limit)

    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {image_dir}")

    draw_gt = args.draw_gt
    draw_pred = args.draw_pred

    if not draw_gt and not draw_pred:
        draw_gt = True
        draw_pred = True

    for feature_path in feature_files:
        feature = load_feature(feature_path)
        rgb = make_rgb(feature, channel_indices, args.percentile_low, args.percentile_high)
        visual = rgb.copy()

        gt_labels = []
        pred_labels = []

        if draw_gt:
            gt_label_path = gt_label_dir / f"{feature_path.stem}.txt"
            gt_labels = read_yolo_labels(gt_label_path, has_conf=False, conf_threshold=0.0)
            visual = draw_boxes(
                image=visual,
                labels=gt_labels,
                color=(0, 255, 0),
                thickness=args.box_thickness,
                prefix="GT",
                show_conf=False,
            )

        if draw_pred:
            pred_label_path = args.pred_label_dir / f"{feature_path.stem}.txt"
            pred_labels = read_yolo_labels(pred_label_path, has_conf=True, conf_threshold=args.conf_threshold)
            visual = draw_boxes(
                image=visual,
                labels=pred_labels,
                color=(255, 0, 0),
                thickness=args.box_thickness,
                prefix="P",
                show_conf=args.show_conf,
            )

            if args.draw_centers:
                visual = draw_centers(
                    image=visual,
                    labels=pred_labels,
                    color=center_color,
                    radius=args.center_radius,
                    prefix="C",
                    show_conf=args.show_conf,
                )

        output_path = output_dir / f"{feature_path.stem}_{args.channels.replace(',', '-')}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
        print(f"[OK] {feature_path.name}: gt={len(gt_labels)}, pred={len(pred_labels)} -> {output_path}")


if __name__ == "__main__":
    main()