import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np


TILE_RE = re.compile(r"r(\d+)_c(\d+)")


def parse_tile_order(path: Path):
    match = TILE_RE.search(path.stem)
    if match is None:
        return 10**9, 10**9
    return int(match.group(1)), int(match.group(2))


def load_tile_points(txt_path: Path):
    try:
        data = np.loadtxt(txt_path)
    except ValueError:
        data = np.loadtxt(txt_path, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 10:
        raise ValueError(
            f"{txt_path} should have at least 10 columns: x y z r g b gt_sem gt_ins pred_sem pred_ins"
        )

    return data[:, :10].astype(np.float64)


def instance_bbox(points_xy):
    x_min = float(np.min(points_xy[:, 0]))
    x_max = float(np.max(points_xy[:, 0]))
    y_min = float(np.min(points_xy[:, 1]))
    y_max = float(np.max(points_xy[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def bbox_intersection(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def smaller_instance_overlap(bbox_a, bbox_b):
    inter = bbox_intersection(bbox_a, bbox_b)
    if inter <= 0.0:
        return 0.0
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def format_output_rows(rows):
    if not rows:
        return np.empty((0, 8), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def save_points(output_path: Path, points):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ["%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.0f", "%.0f"]
    np.savetxt(output_path, points, fmt=fmt)


def merge_tiles(input_dir: Path, output_dir: Path, semantic_id: int, merge_threshold: float, pattern: str):
    txt_files = sorted(input_dir.glob(pattern), key=parse_tile_order)
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {input_dir} with pattern {pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "merge_cache.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        "CREATE TABLE merged_points ("
        "x REAL, y REAL, z REAL, r REAL, g REAL, b REAL, pred_sem INTEGER, global_id INTEGER, "
        "PRIMARY KEY (x, y, z, r, g, b, pred_sem))"
    )

    global_instances = []
    next_global_id = 0

    for idx, txt_path in enumerate(txt_files, start=1):
        data = load_tile_points(txt_path)
        pred_sem = data[:, 8].astype(np.int64)
        pred_ins = data[:, 9].astype(np.int64)
        global_ids = np.full(data.shape[0], -1, dtype=np.int64)

        fg_mask = (pred_sem == semantic_id) & (pred_ins >= 0)
        if np.any(fg_mask):
            local_ids = np.sort(np.unique(pred_ins[fg_mask]))
            for local_id in local_ids:
                local_mask = fg_mask & (pred_ins == local_id)
                local_bbox = instance_bbox(data[local_mask][:, :2])
                best_global_id = None
                best_score = 0.0

                for global_item in global_instances:
                    score = smaller_instance_overlap(local_bbox, global_item["bbox"])
                    if score > best_score:
                        best_score = score
                        best_global_id = global_item["id"]

                if best_global_id is None or best_score < merge_threshold:
                    best_global_id = next_global_id
                    next_global_id += 1
                    global_instances.append({"id": best_global_id, "bbox": local_bbox, "count": int(np.sum(local_mask))})
                else:
                    global_item = next(item for item in global_instances if item["id"] == best_global_id)
                    x_min1, y_min1, x_max1, y_max1 = global_item["bbox"]
                    x_min2, y_min2, x_max2, y_max2 = local_bbox
                    global_item["bbox"] = (min(x_min1, x_min2), min(y_min1, y_min2), max(x_max1, x_max2), max(y_max1, y_max2))
                    global_item["count"] += int(np.sum(local_mask))

                global_ids[local_mask] = best_global_id

        rows = (
            (
                round(float(data[i, 0]), 6),
                round(float(data[i, 1]), 6),
                round(float(data[i, 2]), 6),
                round(float(data[i, 3]), 6),
                round(float(data[i, 4]), 6),
                round(float(data[i, 5]), 6),
                int(pred_sem[i]),
                int(global_ids[i]),
            )
            for i in range(data.shape[0])
        )
        conn.executemany("INSERT OR IGNORE INTO merged_points VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        print(f"[{idx}/{len(txt_files)}] {txt_path.name} merged")

    merged_path = output_dir / "merged_block_points.txt"
    merged_count = 0
    with open(merged_path, "w", encoding="utf-8") as file:
        cursor = conn.execute("SELECT x, y, z, r, g, b, pred_sem, global_id FROM merged_points")
        for row in cursor:
            file.write("%.6f %.6f %.6f %.6f %.6f %.6f %.0f %.0f\n" % row)
            merged_count += 1

    summary_path = output_dir / "merge_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"tiles={len(txt_files)}\n")
        f.write(f"global_instances={len(global_instances)}\n")
        f.write(f"merged_points={merged_count}\n")
        f.write(f"semantic_id={semantic_id}\n")
        f.write(f"merge_threshold={merge_threshold}\n")
        f.write(f"cache_db={db_path}\n")

    conn.close()
    print(f"[OK] merged file: {merged_path}")
    print(f"[OK] summary    : {summary_path}")
    print(f"[OK] instances  : {len(global_instances)}")


def main():
    parser = argparse.ArgumentParser(description="Merge tile-level clustered point clouds into one block with global instance IDs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="输入滑块结果目录，例如 two_stage_clustered_points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出合并目录",
    )
    parser.add_argument(
        "--semantic-id",
        type=int,
        default=1,
        help="需要参与实例融合的前景语义类别ID",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.5,
        help="交集 / 较小实例面积 的合并阈值",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="输入文件匹配模式",
    )
    args = parser.parse_args()

    merge_tiles(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        semantic_id=args.semantic_id,
        merge_threshold=args.merge_threshold,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()