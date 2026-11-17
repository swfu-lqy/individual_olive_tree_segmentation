import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CHANNEL_NAMES = ["VD", "CHM", "G_CHM", "LBP_CHM"]


def read_world_boxes(box_path):
    if not box_path.exists() or box_path.stat().st_size == 0:
        return []

    data = np.genfromtxt(box_path, names=True, dtype=None, encoding="utf-8")

    if data.size == 0:
        return []

    data = np.atleast_1d(data)
    boxes = []

    for row in data:
        boxes.append(
            {
                "pred_instance_id": int(row["pred_instance_id"]),
                "cls_id": int(row["cls_id"]),
                "confidence": float(row["confidence"]),
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
            }
        )

    return boxes


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_feature(feature_path):
    feature = np.load(feature_path)
    if feature.ndim != 3:
        raise ValueError(f"{feature_path} should be 3D")
    if feature.shape[0] == 4:
        return feature
    if feature.shape[-1] == 4:
        return np.transpose(feature, (2, 0, 1))
    raise ValueError(f"{feature_path} should have 4 channels, got {feature.shape}")


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
    channels = [normalize_channel(feature[index], low_percentile, high_percentile) for index in channel_indices]
    return np.stack(channels, axis=-1)


def world_to_pixel(x, y, metadata, image_size):
    image_w, image_h = image_size
    px = int(round((x - float(metadata["min_x"])) / float(metadata["resolution"])))
    py = int(round((float(metadata["max_y"]) - y) / float(metadata["resolution"])))
    px = max(0, min(px, image_w - 1))
    py = max(0, min(py, image_h - 1))
    return px, py


def draw_world_boxes(base_image, boxes, metadata, box_color, center_color, box_thickness, center_radius, show_id, show_conf):
    canvas = base_image.copy()
    image_h, image_w = canvas.shape[:2]
    image_size = (image_w, image_h)

    if not boxes:
        return canvas

    for index, box in enumerate(boxes):
        x1, y1 = world_to_pixel(box["x_min"], box["y_min"], metadata, image_size)
        x2, y2 = world_to_pixel(box["x_max"], box["y_max"], metadata, image_size)
        cx, cy = world_to_pixel(box["center_x"], box["center_y"], metadata, image_size)

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        cv2.rectangle(canvas, (left, top), (right, bottom), box_color, box_thickness)
        cv2.circle(canvas, (cx, cy), center_radius, center_color, -1, cv2.LINE_AA)

        text_items = []
        if show_id:
            text_items.append(f"ID:{box['pred_instance_id']}")
        if show_conf:
            text_items.append(f"{box['confidence']:.2f}")

        if text_items:
            text = " ".join(text_items)
            text_x = min(image_w - 1, cx + center_radius + 3)
            text_y = max(12, cy - center_radius - 3)
            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                center_color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def parse_color(color_text):
    values = [int(value.strip()) for value in color_text.split(",")]
    if len(values) != 3:
        raise ValueError("颜色参数必须是 3 个逗号分隔的整数，例如 255,0,0")
    return tuple(max(0, min(value, 255)) for value in values)


def parse_channels(channels):
    values = [value.strip() for value in channels.split(",")]
    if len(values) != 3:
        raise ValueError("--channels 必须包含 3 个通道，例如 CHM,G_CHM,LBP_CHM")
    indices = []
    for value in values:
        if value.isdigit():
            index = int(value)
        else:
            if value not in CHANNEL_NAMES:
                raise ValueError(f"未知通道名称: {value}")
            index = CHANNEL_NAMES.index(value)
        if index < 0 or index >= 4:
            raise ValueError(f"通道索引越界: {index}")
        indices.append(index)
    return indices


def get_block_name(box_path):
    suffix = "_pred_boxes_world"
    stem = box_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def collect_box_files(box_dir, pattern, limit):
    box_files = sorted(box_dir.glob(pattern))
    if limit > 0:
        box_files = box_files[:limit]
    return box_files


def main():
    parser = argparse.ArgumentParser(description="读取映射到点云坐标系后的预测框，并在二维平面图上绘制边界框和中心点")
    parser.add_argument("--box-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/yolo_maped_points_boxes"), help="过滤后的世界坐标预测框 txt 文件目录")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/olive_data_yolo_format"), help="YOLO 格式 4 通道数据集根目录，脚本会读取 images/{split} 下的 .npy 作为底图")
    parser.add_argument("--feature-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/olive_data_semantic_predicted_4ch_features"), help="包含各地块 metadata.json 的 4 通道特征目录")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="需要读取的底图数据划分")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/yolo_maped_points_boxes_visual"), help="二维框体可视化图片输出目录")
    parser.add_argument("--pattern", type=str, default="*_pred_boxes_world.txt", help="需要读取的预测框文件匹配模式")
    parser.add_argument("--channels", type=str, default="CHM,G_CHM,LBP_CHM", help="用于合成 RGB 底图的 3 个通道名称或索引，逗号分隔")
    parser.add_argument("--percentile-low", type=float, default=1.0, help="通道归一化低百分位截断值")
    parser.add_argument("--percentile-high", type=float, default=99.0, help="通道归一化高百分位截断值")
    parser.add_argument("--box-color", type=str, default="255,0,0", help="边界框颜色，格式为 B,G,R，例如 255,0,0 表示蓝色")
    parser.add_argument("--center-color", type=str, default="0,0,0", help="中心点颜色，格式为 B,G,R，例如 0,0,0 表示黑色")
    parser.add_argument("--box-thickness", type=int, default=2, help="边界框线宽，单位为像素")
    parser.add_argument("--center-radius", type=int, default=5, help="中心点半径，单位为像素")
    parser.add_argument("--show-id", action="store_true", help="是否在中心点旁显示预测实例 ID")
    parser.add_argument("--show-conf", action="store_true", help="是否在中心点旁显示预测框置信度")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个地块文件，0 表示不限制")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    box_color = parse_color(args.box_color)
    center_color = parse_color(args.center_color)
    channel_indices = parse_channels(args.channels)

    box_files = collect_box_files(args.box_dir, args.pattern, args.limit)

    if not box_files:
        raise FileNotFoundError(f"No box files found in {args.box_dir} with pattern {args.pattern}")

    for box_path in box_files:
        block_name = get_block_name(box_path)
        feature_path = args.dataset_dir / "images" / args.split / f"{block_name}.npy"
        metadata_path = args.feature_dir / block_name / f"{block_name}_metadata.json"

        if not feature_path.exists():
            print(f"[SKIP] {block_name}: 底图不存在 -> {feature_path}")
            continue
        if not metadata_path.exists():
            print(f"[SKIP] {block_name}: metadata 不存在 -> {metadata_path}")
            continue

        boxes = read_world_boxes(box_path)
        feature = load_feature(feature_path)
        base_image = make_rgb(feature, channel_indices, args.percentile_low, args.percentile_high)
        metadata = load_metadata(metadata_path)
        canvas = draw_world_boxes(
            base_image=base_image,
            boxes=boxes,
            metadata=metadata,
            box_color=box_color,
            center_color=center_color,
            box_thickness=args.box_thickness,
            center_radius=args.center_radius,
            show_id=args.show_id,
            show_conf=args.show_conf,
        )

        output_path = args.output_dir / f"{box_path.stem}_2d_boxes.png"
        cv2.imwrite(str(output_path), canvas)

        print(f"[OK] {box_path.name}: boxes={len(boxes)} -> {output_path}")


if __name__ == "__main__":
    main()