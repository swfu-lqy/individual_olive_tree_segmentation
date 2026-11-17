import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _parse_column_spec(column_spec, available_columns):
    import re
    indices = []
    for token in column_spec.split(","):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"-?\d+", token):
            idx = int(token)
            if idx < 0 or idx >= available_columns:
                raise ValueError(f"Column index {idx} out of range (0..{available_columns - 1})")
            indices.append(idx)
        else:
            if isinstance(available_columns, int):
                raise ValueError(f"Cannot resolve column name '{token}' without a header row (add --has-header)")
            if token not in available_columns:
                raise ValueError(f"Column name '{token}' not found. Available: {available_columns}")
            indices.append(available_columns.index(token))
    if not indices:
        raise ValueError("Empty column specification")
    return indices


def _read_header(header_line):
    return [col.strip() for col in header_line.strip().split(",")] if "," in header_line else header_line.strip().split()


def read_point_cloud(txt_path, column_spec=None, has_header=False):
    with open(txt_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            raise ValueError(f"{txt_path} is empty")

    if "," in first_line:
        detected_delimiter = ","
    else:
        detected_delimiter = None

    if has_header:
        available_columns = _read_header(first_line) if detected_delimiter is None else [c.strip() for c in first_line.strip().split(",")]
    else:
        first_line_stripped = first_line.strip()
        if detected_delimiter == ",":
            available_columns = len([v for v in first_line_stripped.split(",") if v != ""])
        else:
            available_columns = len(first_line_stripped.split())

    spec = column_spec or "x,y,z,semantic_id"
    columns = _parse_column_spec(spec, available_columns)

    with open(txt_path, "r", encoding="utf-8") as file:
        if has_header:
            file.readline()
        raw = np.loadtxt(file, delimiter=detected_delimiter)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1) if raw.size > 0 else raw.reshape(0, 0)

    if raw.shape[1] != available_columns:
        raise ValueError(
            f"{txt_path}: 首行与正文列数不一致 (首行={available_columns}, 实际={raw.shape[1]})."
            f"分隔符={detected_delimiter!r}。"
        )

    keep = [c for c in columns if c < raw.shape[1]]
    data = raw[:, keep]

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(
            f"{txt_path}: 实际读到 {data.shape[1]} 列 (期望至少 4 列)。"
            f"文件分隔符嗅探结果={detected_delimiter!r}, 输入列数={available_columns}, --columns={column_spec!r}"
        )
    return data


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def xy_to_pixel(x, y, metadata):
    min_x = metadata["min_x"]
    max_y = metadata["max_y"]
    resolution = metadata["resolution"]

    col = np.floor((x - min_x) / resolution).astype(np.float64)
    row = np.floor((max_y - y) / resolution).astype(np.float64)

    return col, row


def instance_boxes_to_yolo_labels(points, metadata, semantic_id, min_points, min_box_size, has_instance_id=True, column_x=0, column_y=1, column_semantic=2, column_instance=3):
    olive = points[points[:, column_semantic].astype(np.int64) == semantic_id]
    if has_instance_id:
        if column_instance >= points.shape[1]:
            raise ValueError(
                f"points only has {points.shape[1]} columns, cannot read instance_id at column {column_instance}. "
                f"请在 --columns 中添加 instance_id 列，例如 --columns \"0,1,2,3,4\""
            )
        olive = olive[olive[:, column_instance].astype(np.int64) >= 0]

    if olive.shape[0] == 0:
        return []

    image_h = int(metadata["height"])
    image_w = int(metadata["width"])

    labels = []

    if has_instance_id:
        instance_ids = np.unique(olive[:, column_instance].astype(np.int64))
        groups = [(instance_id, olive[olive[:, column_instance].astype(np.int64) == instance_id]) for instance_id in instance_ids]
    else:
        groups = [("all", olive)]

    for instance_id, instance_points in groups:
        if instance_points.shape[0] < min_points:
            continue

        col, row = xy_to_pixel(instance_points[:, column_x], instance_points[:, column_y], metadata)

        x1 = float(np.min(col))
        x2 = float(np.max(col))
        y1 = float(np.min(row))
        y2 = float(np.max(row))

        x1 = max(0.0, min(x1, image_w - 1.0))
        x2 = max(0.0, min(x2, image_w - 1.0))
        y1 = max(0.0, min(y1, image_h - 1.0))
        y2 = max(0.0, min(y2, image_h - 1.0))

        box_w = x2 - x1 + 1.0
        box_h = y2 - y1 + 1.0

        if box_w < min_box_size or box_h < min_box_size:
            continue

        x_center = ((x1 + x2) / 2.0) / image_w
        y_center = ((y1 + y2) / 2.0) / image_h
        norm_w = box_w / image_w
        norm_h = box_h / image_h

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return labels


def find_feature_file(feature_dir, stem):
    candidates = [
        feature_dir / stem / f"{stem}_4ch.npy",
        feature_dir / stem / f"{stem}_features.npz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No 4ch feature found for {stem}")


def copy_feature_to_yolo_image(feature_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_path.suffix == ".npy":
        shutil.copy2(feature_path, output_path)
        return

    data = np.load(feature_path)
    feature = data["feature_4ch"]
    np.save(output_path, feature.astype(np.float32))


def write_yaml(output_dir):
    yaml_path = output_dir / "olive_4ch.yaml"
    content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: olive_tree
channels: 4
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def process_split(split, point_dir, feature_dir, output_dir, semantic_id, min_points, min_box_size, has_instance_id=True, column_spec=None, has_header=False):
    role_indices = _resolve_role_indices(point_dir, column_spec, has_header, has_instance_id)

    role_to_kept = {role: i for i, role in enumerate(["x", "y", "z", "semantic_id"] + (["instance_id"] if has_instance_id else []))}

    txt_files = sorted(point_dir.glob("*.txt"))

    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    count_images = 0
    count_boxes = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        metadata_path = feature_dir / stem / f"{stem}_metadata.json"

        if not metadata_path.exists():
            print(f"[SKIP] {stem}: metadata not found, feature may not have been generated")
            continue

        metadata = load_metadata(metadata_path)
        if metadata.get("status") == "no_olive_points" or int(metadata.get("olive_point_count", 0)) == 0:
            print(f"[SKIP] {stem}: no olive points")
            continue

        try:
            feature_path = find_feature_file(feature_dir, stem)
        except FileNotFoundError as error:
            print(f"[SKIP] {stem}: {error}")
            continue

        points = read_point_cloud(txt_path, column_spec=column_spec, has_header=has_header)

        labels = instance_boxes_to_yolo_labels(
            points=points,
            metadata=metadata,
            semantic_id=semantic_id,
            min_points=min_points,
            min_box_size=min_box_size,
            has_instance_id=has_instance_id,
            column_x=role_to_kept["x"],
            column_y=role_to_kept["y"],
            column_semantic=role_to_kept["semantic_id"],
            column_instance=role_to_kept.get("instance_id", 0),
        )

        output_image_path = image_dir / f"{stem}.npy"
        output_label_path = label_dir / f"{stem}.txt"

        copy_feature_to_yolo_image(feature_path, output_image_path)
        output_label_path.write_text("\n".join(labels), encoding="utf-8")

        count_images += 1
        count_boxes += len(labels)

    print(f"[{split}] has_instance_id={has_instance_id} images={count_images}, boxes={count_boxes}")


def _resolve_role_indices(point_dir, column_spec, has_header, has_instance_id):
    sample_files = sorted(point_dir.glob("*.txt"))
    if not sample_files:
        raise FileNotFoundError(f"No .txt files found in {point_dir}")
    sample_path = sample_files[0]

    with open(sample_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
    if has_header:
        available_columns = _read_header(first_line)
    else:
        if "," in first_line:
            available_columns = len(first_line.strip().split(","))
        else:
            available_columns = len(first_line.strip().split())

    selected = _parse_column_spec(column_spec or "x,y,z,semantic_id", available_columns)

    role_names = ["x", "y", "z", "semantic_id"]
    if has_instance_id:
        role_names.append("instance_id")

    if len(selected) < len(role_names):
        if has_instance_id:
            raise ValueError(
                f"--columns must provide at least 5 columns (x,y,z,semantic_id,instance_id) when --has-instance-id is set, got {len(selected)}: {column_spec}"
            )
        raise ValueError(
            f"--columns must provide at least 4 columns (x,y,z,semantic_id), got {len(selected)}: {column_spec}"
        )

    return {role: selected[i] for i, role in enumerate(role_names)}


def main():
    parser = argparse.ArgumentParser(
        description="根据 4 通道特征与点云 instance_id，制作 YOLO 格式橄榄树冠检测数据集 (images/*.npy + labels/*.txt)",
    )
    parser.add_argument("--train-point-dir", type=Path, default=None,
                        help="训练集点云目录 (传 None 表示不制作训练集)")
    parser.add_argument("--train-feature-dir", type=Path, default=None,
                        help="训练集 4ch 特征目录，需与 --train-point-dir 配对")
    parser.add_argument("--val-point-dir", type=Path, default=None,
                        help="验证集点云目录 (传 None 表示不制作验证集)")
    parser.add_argument("--val-feature-dir", type=Path, default=None,
                        help="验证集 4ch 特征目录，需与 --val-point-dir 配对")
    parser.add_argument("--test-point-dir", type=Path, default=None,
                        help="测试集点云目录 (传 None 表示不制作测试集)")
    parser.add_argument("--test-feature-dir", type=Path, default=None,
                        help="测试集 4ch 特征目录，需与 --test-point-dir 配对")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/yolo_olive_4ch"),
                        help="YOLO 数据集输出根目录")
    parser.add_argument("--semantic-id", type=int, default=1,
                        help="语义类别 ID，表示橄榄树点 (默认 1)")
    parser.add_argument("--min-points", type=int, default=30,
                        help="单个实例的最小点数，少于这个点数的框会被丢弃 (默认 30)")
    parser.add_argument("--min-box-size", type=float, default=2.0,
                        help="边框最小边长 (像素)，低于此的边框会被丢弃 (默认 2.0)")
    parser.add_argument("--columns", type=str, default="x,y,z,semantic_id",
                        help="需要读取的列，逗号分隔，支持列名 (如 x) 或列号 (如 0,1,2,6,7)")
    parser.add_argument("--no-header", dest="has_header", action="store_false",
                        help="声明输入文件没有表头行 (默认)")
    parser.add_argument("--has-header", dest="has_header", action="store_true",
                        help="声明输入文件首行是表头行")
    parser.set_defaults(has_header=False)
    parser.add_argument("--no-instance-id", dest="has_instance_id", action="store_false",
                        help="声明点云没有 instance_id (例如预测场景)：按整个 olive 区域画一个框")
    parser.add_argument("--has-instance-id", dest="has_instance_id", action="store_true",
                        help="声明点云有 instance_id (训练场景)：按 instance 聚类画框")
    parser.set_defaults(has_instance_id=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    enabled_splits = []
    if args.train_point_dir is not None and args.train_feature_dir is not None:
        if not args.train_point_dir.exists() or not args.train_feature_dir.exists():
            raise FileNotFoundError(f"train point/feature dir not found: {args.train_point_dir} / {args.train_feature_dir}")
        enabled_splits.append("train")
    if args.val_point_dir is not None and args.val_feature_dir is not None:
        if not args.val_point_dir.exists() or not args.val_feature_dir.exists():
            raise FileNotFoundError(f"val point/feature dir not found: {args.val_point_dir} / {args.val_feature_dir}")
        enabled_splits.append("val")
    if args.test_point_dir is not None and args.test_feature_dir is not None:
        if not args.test_point_dir.exists() or not args.test_feature_dir.exists():
            raise FileNotFoundError(f"test point/feature dir not found: {args.test_point_dir} / {args.test_feature_dir}")
        enabled_splits.append("test")

    if not enabled_splits:
        raise SystemExit(
            "未启用任何 split: 请至少提供一组 (--*-point-dir + --*-feature-dir)。\n"
            "例如:\n"
            "  # 仅测试 (无 instance_id)\n"
            "  python build_olive_yolo_dataset.py --no-instance-id \\\n"
            "      --test-point-dir ... --test-feature-dir ...\n"
            "  # 仅训练 (train+val 有 instance_id)\n"
            "  python build_olive_yolo_dataset.py \\\n"
            "      --train-point-dir ... --train-feature-dir ... \\\n"
            "      --val-point-dir ... --val-feature-dir ..."
        )

    print(f"[INFO] enabled splits = {enabled_splits}  has_instance_id={args.has_instance_id}  columns={args.columns}")

    if "train" in enabled_splits:
        process_split("train", args.train_point_dir, args.train_feature_dir, args.output_dir,
                      args.semantic_id, args.min_points, args.min_box_size,
                      has_instance_id=args.has_instance_id, column_spec=args.columns, has_header=args.has_header)
    if "val" in enabled_splits:
        process_split("val", args.val_point_dir, args.val_feature_dir, args.output_dir,
                      args.semantic_id, args.min_points, args.min_box_size,
                      has_instance_id=args.has_instance_id, column_spec=args.columns, has_header=args.has_header)
    if "test" in enabled_splits:
        process_split("test", args.test_point_dir, args.test_feature_dir, args.output_dir,
                      args.semantic_id, args.min_points, args.min_box_size,
                      has_instance_id=args.has_instance_id, column_spec=args.columns, has_header=args.has_header)

    yaml_path = write_yaml(args.output_dir)
    print(f"[OK] dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()