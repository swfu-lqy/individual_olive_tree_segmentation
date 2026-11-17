import argparse
import json
from pathlib import Path

import numpy as np


def read_point_cloud(txt_path):
    try:
        data = np.loadtxt(txt_path)
    except ValueError:
        data = np.loadtxt(txt_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 9:
        raise ValueError(f"{txt_path} should have at least 9 columns: x y z r g b gt_semantic gt_instance pred_semantic")
    return data


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_prediction_labels(label_path, conf_threshold):
    if not label_path.exists():
        return []

    boxes = []
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
        confidence = float(parts[5]) if len(parts) >= 6 else 1.0

        if confidence < conf_threshold:
            continue

        boxes.append(
            {
                "cls_id": cls_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "confidence": confidence,
            }
        )

    boxes.sort(key=lambda item: item["confidence"], reverse=True)
    return boxes


def box_area(box):
    return max(0.0, box["width"]) * max(0.0, box["height"])


def box_iou(box_a, box_b):
    ax1 = box_a["x_center"] - box_a["width"] / 2.0
    ay1 = box_a["y_center"] - box_a["height"] / 2.0
    ax2 = box_a["x_center"] + box_a["width"] / 2.0
    ay2 = box_a["y_center"] + box_a["height"] / 2.0

    bx1 = box_b["x_center"] - box_b["width"] / 2.0
    by1 = box_b["y_center"] - box_b["height"] / 2.0
    bx2 = box_b["x_center"] + box_b["width"] / 2.0
    by2 = box_b["y_center"] + box_b["height"] / 2.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = box_area(box_a) + box_area(box_b) - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def filter_small_boxes(boxes, min_area_ratio):
    if min_area_ratio <= 0.0 or len(boxes) <= 1:
        return boxes

    areas = np.array([box_area(box) for box in boxes], dtype=np.float64)
    positive_areas = areas[areas > 0.0]
    if positive_areas.size == 0:
        return []

    min_area = float(np.median(positive_areas) * min_area_ratio)
    return [box for box, area in zip(boxes, areas) if area >= min_area]


def center_distance(box_a, box_b):
    return float(np.sqrt((box_a["x_center"] - box_b["x_center"]) ** 2 + (box_a["y_center"] - box_b["y_center"]) ** 2))


def calculate_local_spacing_threshold(boxes, stat_method):
    if len(boxes) < 3:
        return None

    local_spacings = []
    for index, box in enumerate(boxes):
        distances = []
        for other_index, other_box in enumerate(boxes):
            if index == other_index:
                continue
            distances.append(center_distance(box, other_box))
        distances.sort()
        if len(distances) >= 2:
            local_spacings.append(max(distances[0], distances[1]))

    if not local_spacings:
        return None

    local_spacings = np.array(local_spacings, dtype=np.float64)
    if stat_method == "mean":
        return float(np.mean(local_spacings))
    if stat_method == "median":
        return float(np.median(local_spacings))
    if stat_method == "min":
        return float(np.min(local_spacings))
    raise ValueError(f"Unknown center distance stat method: {stat_method}")


def filter_close_center_boxes(boxes, enabled, stat_method, threshold_scale):
    if not enabled or len(boxes) < 3:
        return boxes

    threshold = calculate_local_spacing_threshold(boxes, stat_method)
    if threshold is None or threshold <= 0.0:
        return boxes

    threshold *= threshold_scale
    pairs = []
    for index, box in enumerate(boxes):
        nearest_index = None
        nearest_distance = None
        for other_index, other_box in enumerate(boxes):
            if index == other_index or box["cls_id"] != other_box["cls_id"]:
                continue
            distance = center_distance(box, other_box)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_index = other_index
        if nearest_index is not None and nearest_distance < threshold:
            pair = tuple(sorted((index, nearest_index)))
            pairs.append((nearest_distance, pair[0], pair[1]))

    removed_indices = set()
    unique_pairs = sorted(set(pairs), key=lambda item: item[0])
    for distance, index_a, index_b in unique_pairs:
        if index_a in removed_indices or index_b in removed_indices:
            continue
        area_a = box_area(boxes[index_a])
        area_b = box_area(boxes[index_b])
        if area_a < area_b:
            removed_indices.add(index_a)
        elif area_b < area_a:
            removed_indices.add(index_b)
        elif boxes[index_a]["confidence"] < boxes[index_b]["confidence"]:
            removed_indices.add(index_a)
        else:
            removed_indices.add(index_b)

    return [box for index, box in enumerate(boxes) if index not in removed_indices]


def filter_overlapping_boxes(boxes, overlap_threshold):
    if overlap_threshold <= 0.0 or len(boxes) <= 1:
        return boxes

    kept_boxes = []
    for box in boxes:
        duplicate = False
        for kept_box in kept_boxes:
            if box["cls_id"] != kept_box["cls_id"]:
                continue
            if box_iou(box, kept_box) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept_boxes.append(box)

    return kept_boxes


def filter_prediction_boxes(boxes, overlap_threshold, min_area_ratio, center_distance_filter, center_distance_stat, center_distance_scale):
    filtered_boxes = filter_small_boxes(boxes, min_area_ratio)
    filtered_boxes = filter_close_center_boxes(filtered_boxes, center_distance_filter, center_distance_stat, center_distance_scale)
    return filter_overlapping_boxes(filtered_boxes, overlap_threshold)


def yolo_box_to_world_box(box, metadata):
    image_w = float(metadata["width"])
    image_h = float(metadata["height"])
    min_x = float(metadata["min_x"])
    max_y = float(metadata["max_y"])
    resolution = float(metadata["resolution"])

    x_center_px = box["x_center"] * image_w
    y_center_px = box["y_center"] * image_h
    box_w_px = box["width"] * image_w
    box_h_px = box["height"] * image_h

    x1_px = max(0.0, x_center_px - box_w_px / 2.0)
    x2_px = min(image_w - 1.0, x_center_px + box_w_px / 2.0)
    y1_px = max(0.0, y_center_px - box_h_px / 2.0)
    y2_px = min(image_h - 1.0, y_center_px + box_h_px / 2.0)

    x1_world = min_x + x1_px * resolution
    x2_world = min_x + x2_px * resolution

    y_top_world = max_y - y1_px * resolution
    y_bottom_world = max_y - y2_px * resolution

    x_min = min(x1_world, x2_world)
    x_max = max(x1_world, x2_world)
    y_min = min(y_top_world, y_bottom_world)
    y_max = max(y_top_world, y_bottom_world)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_x": center_x,
        "center_y": center_y,
        "confidence": box["confidence"],
        "cls_id": box["cls_id"],
    }


def build_world_boxes(pred_boxes, metadata):
    world_boxes = []
    for pred_instance_id, box in enumerate(pred_boxes):
        world_box = yolo_box_to_world_box(box, metadata)
        world_box["pred_instance_id"] = pred_instance_id
        world_boxes.append(world_box)
    return world_boxes


def assign_points_to_boxes(points, world_boxes, semantic_id):
    assigned_points = np.hstack([points, np.full((points.shape[0], 1), -1.0, dtype=np.float64)])

    if points.shape[0] == 0 or len(world_boxes) == 0:
        return assigned_points

    semantic_mask = points[:, 8].astype(np.int64) == semantic_id
    if not np.any(semantic_mask):
        return assigned_points

    box_arrays = np.array(
        [
            [
                box["x_min"],
                box["x_max"],
                box["y_min"],
                box["y_max"],
                box["center_x"],
                box["center_y"],
                box["pred_instance_id"],
                box["confidence"],
            ]
            for box in world_boxes
        ],
        dtype=np.float64,
    )

    candidate_indices = np.where(semantic_mask)[0]
    for point_index in candidate_indices:
        point = points[point_index]
        x = point[0]
        y = point[1]

        inside = (
            (x >= box_arrays[:, 0])
            & (x <= box_arrays[:, 1])
            & (y >= box_arrays[:, 2])
            & (y <= box_arrays[:, 3])
        )

        matched_indices = np.where(inside)[0]

        if matched_indices.size == 0:
            continue

        if matched_indices.size == 1:
            assigned_instance_id = int(box_arrays[matched_indices[0], 6])
        else:
            confidences = box_arrays[matched_indices, 7]
            best_index = matched_indices[int(np.argmax(confidences))]
            assigned_instance_id = int(box_arrays[best_index, 6])

        assigned_points[point_index, -1] = assigned_instance_id

    return assigned_points


def save_world_boxes(world_boxes, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("pred_instance_id cls_id confidence x_min y_min x_max y_max center_x center_y\n")
        for box in world_boxes:
            file.write(
                f"{box['pred_instance_id']} "
                f"{box['cls_id']} "
                f"{box['confidence']:.6f} "
                f"{box['x_min']:.6f} "
                f"{box['y_min']:.6f} "
                f"{box['x_max']:.6f} "
                f"{box['y_max']:.6f} "
                f"{box['center_x']:.6f} "
                f"{box['center_y']:.6f}\n"
            )


def save_assigned_points(points, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] == 0:
        output_path.write_text("", encoding="utf-8")
        return

    if points.shape[1] == 9:
        fmt = ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"]
    elif points.shape[1] == 10:
        fmt = ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"]
    else:
        fmt = ["%.6f"] * 3 + ["%.0f"] * (points.shape[1] - 3)
    np.savetxt(output_path, points, fmt=fmt)


def process_file(
    txt_path,
    feature_dir,
    pred_label_dir,
    output_point_dir,
    output_box_dir,
    semantic_id,
    conf_threshold,
    overlap_threshold,
    min_area_ratio,
    center_distance_filter,
    center_distance_stat,
    center_distance_scale,
):
    stem = txt_path.stem

    metadata_path = feature_dir / stem / f"{stem}_metadata.json"
    pred_label_path = pred_label_dir / f"{stem}.txt"

    if not metadata_path.exists():
        print(f"[SKIP] {stem}: metadata not found")
        return

    metadata = load_metadata(metadata_path)

    if metadata.get("status") == "no_olive_points" or int(metadata.get("olive_point_count", 0)) == 0:
        print(f"[SKIP] {stem}: no olive points")
        return

    pred_boxes = read_prediction_labels(pred_label_path, conf_threshold)
    original_box_count = len(pred_boxes)
    pred_boxes = filter_prediction_boxes(
        pred_boxes,
        overlap_threshold,
        min_area_ratio,
        center_distance_filter,
        center_distance_stat,
        center_distance_scale,
    )

    if not pred_boxes:
        output_point_path = output_point_dir / f"{stem}_pred_instances.txt"
        output_box_path = output_box_dir / f"{stem}_pred_boxes_world.txt"
        points = read_point_cloud(txt_path)
        assigned_points = assign_points_to_boxes(points, [], semantic_id)
        save_assigned_points(assigned_points, output_point_path)
        save_world_boxes([], output_box_path)
        print(f"[EMPTY] {stem}: no predictions after conf/filter, saved all points with pred_instance_id=-1, conf >= {conf_threshold}")
        return

    world_boxes = build_world_boxes(pred_boxes, metadata)
    points = read_point_cloud(txt_path)
    assigned_points = assign_points_to_boxes(points, world_boxes, semantic_id)

    output_point_path = output_point_dir / f"{stem}_pred_instances.txt"
    output_box_path = output_box_dir / f"{stem}_pred_boxes_world.txt"

    save_assigned_points(assigned_points, output_point_path)
    save_world_boxes(world_boxes, output_box_path)

    assigned_count = int(np.sum(assigned_points[:, -1] >= 0))
    print(
        f"[OK] {stem}: boxes={len(world_boxes)}/{original_box_count}, assigned_points={assigned_count} "
        f"-> {output_point_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="将 YOLO 预测框转换到点云坐标系，并把预测语义橄榄点分配到预测实例")
    parser.add_argument("--point-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/test"), help="预测语义点云 txt 文件所在目录，文件列至少包含 x,y,z,r,g,b,gt_semantic,gt_instance,pred_semantic")
    parser.add_argument("--feature-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/test_4ch_features"), help="4 通道特征和 metadata.json 所在目录")
    parser.add_argument("--pred-label-dir", type=Path, default=Path("/workspace/Yolo_liqiying/runs_olive_4ch_predict/best_test_pred/labels"), help="YOLO 预测输出 labels 目录，txt 中可包含置信度")
    parser.add_argument("--output-point-dir", type=Path, default=Path("/workspace/Yolo_liqiying/runs_olive_4ch_predict/mapped_points"), help="映射回点云后的实例点输出目录")
    parser.add_argument("--output-box-dir", type=Path, default=Path("/workspace/Yolo_liqiying/runs_olive_4ch_predict/world_boxes"), help="转换到点云坐标系后的预测框输出目录")
    parser.add_argument("--semantic-id", type=int, default=1, help="需要参与实例分配的语义类别 ID")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="预测框置信度阈值，低于该值的框会被忽略")
    parser.add_argument("--overlap-threshold", type=float, default=0.95, help="二次去重的重叠 IoU 阈值，大于等于该值时删除低置信度重叠框，设置为 0 可关闭")
    parser.add_argument("--min-area-ratio", type=float, default=0.30, help="小框过滤阈值，删除面积小于当前场景预测框面积中位数乘以该比例的框，设置为 0 可关闭")
    parser.add_argument("--center-distance-filter", action="store_true", help="是否启用中心点过近过滤，启用后会删除中心点距离过近的一对框中面积较小的框")
    parser.add_argument("--center-distance-stat", choices=["mean", "median", "min"], default="median", help="局部株间距阈值的统计方式，mean 为平均值，median 为中位数，min 为最小值")
    parser.add_argument("--center-distance-scale", type=float, default=1.0, help="局部株间距阈值缩放系数，实际过滤阈值等于统计值乘以该系数")
    parser.add_argument("--pattern", type=str, default="*.txt", help="需要处理的点云文件匹配模式")
    args = parser.parse_args()

    args.output_point_dir.mkdir(parents=True, exist_ok=True)
    args.output_box_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.point_dir.glob(args.pattern))

    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {args.point_dir}")

    for txt_path in txt_files:
        process_file(
            txt_path=txt_path,
            feature_dir=args.feature_dir,
            pred_label_dir=args.pred_label_dir,
            output_point_dir=args.output_point_dir,
            output_box_dir=args.output_box_dir,
            semantic_id=args.semantic_id,
            conf_threshold=args.conf_threshold,
            overlap_threshold=args.overlap_threshold,
            min_area_ratio=args.min_area_ratio,
            center_distance_filter=args.center_distance_filter,
            center_distance_stat=args.center_distance_stat,
            center_distance_scale=args.center_distance_scale,
        )


if __name__ == "__main__":
    main()