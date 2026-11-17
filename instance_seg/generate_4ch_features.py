import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import QhullError

try:
    import CSF
except ImportError:
    CSF = None


CHANNEL_NAMES = ["VD", "CHM", "G_CHM", "LBP_CHM"]


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
    print("读取点云文件：", txt_path)

    with open(txt_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            raise ValueError(f"{txt_path} is empty")

    if has_header:
        available_columns = _read_header(first_line)
    else:
        with open(txt_path, "r", encoding="utf-8") as file:
            first_data = file.readline()
        if "," in first_data:
            first_row = np.array([float(v) for v in first_data.strip().split(",")])
        else:
            first_row = np.array([float(v) for v in first_data.strip().split()])
        available_columns = first_row.shape[0]

    spec = column_spec or "x,y,z,semantic_id"
    columns = _parse_column_spec(spec, available_columns)

    with open(txt_path, "r", encoding="utf-8") as file:
        if has_header:
            file.readline()
        try:
            data = np.loadtxt(file, delimiter=None, usecols=columns)
        except ValueError as e:
            data = np.loadtxt(file, delimiter=",", usecols=columns)



    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(f"{txt_path}: --columns must specify exactly 4 columns (x,y,z,semantic_id), got {data.shape[1]}")
    return data


def split_ground_by_csf(points_xyz, cloth_resolution):
    if CSF is None:
        raise ImportError("CSF is not installed. Please install CSF or use --dtm-method lower_envelope")

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = cloth_resolution
    csf.setPointCloud(points_xyz)

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_index = np.asarray(ground, dtype=np.int64)
    if ground_index.size == 0:
        raise ValueError("CSF did not find ground points")

    return points_xyz[ground_index]


def split_ground_by_lower_envelope(points_xyz, cell_size, percentile):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    min_x = np.min(x)
    min_y = np.min(y)

    col = np.floor((x - min_x) / cell_size).astype(np.int64)
    row = np.floor((y - min_y) / cell_size).astype(np.int64)

    cell_id = row * (np.max(col) + 1) + col
    unique_cells = np.unique(cell_id)

    ground_points = []
    for current_cell in unique_cells:
        mask = cell_id == current_cell
        cell_points = points_xyz[mask]
        z_threshold = np.percentile(cell_points[:, 2], percentile)
        candidate = cell_points[cell_points[:, 2] <= z_threshold]
        if candidate.shape[0] > 0:
            ground_points.append(candidate[np.argmin(candidate[:, 2])])

    if not ground_points:
        raise ValueError("lower_envelope did not find ground points")

    return np.asarray(ground_points, dtype=np.float64)


def build_dtm_interpolator(ground_points, dtm_resolution, method):
    min_x = float(np.min(ground_points[:, 0]))
    max_x = float(np.max(ground_points[:, 0]))
    min_y = float(np.min(ground_points[:, 1]))
    max_y = float(np.max(ground_points[:, 1]))

    x_coords = np.arange(min_x, max_x + dtm_resolution, dtm_resolution)
    y_coords = np.arange(min_y, max_y + dtm_resolution, dtm_resolution)
    if x_coords.size < 2:
        x_coords = np.array([min_x, min_x + dtm_resolution], dtype=np.float64)
    if y_coords.size < 2:
        y_coords = np.array([min_y, min_y + dtm_resolution], dtype=np.float64)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")

    unique_xy_count = np.unique(ground_points[:, :2], axis=0).shape[0]
    effective_method = method if unique_xy_count >= 4 else "nearest"
    try:
        grid_z = griddata(
            ground_points[:, :2],
            ground_points[:, 2],
            (grid_x, grid_y),
            method=effective_method,
        )
    except QhullError:
        effective_method = "nearest"
        grid_z = griddata(
            ground_points[:, :2],
            ground_points[:, 2],
            (grid_x, grid_y),
            method=effective_method,
        )

    if np.isnan(grid_z).any():
        nearest_z = griddata(
            ground_points[:, :2],
            ground_points[:, 2],
            (grid_x, grid_y),
            method="nearest",
        )
        grid_z = np.where(np.isnan(grid_z), nearest_z, grid_z)

    interpolator = RegularGridInterpolator(
        (x_coords, y_coords),
        grid_z,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return interpolator, {
        "dtm_min_x": min_x,
        "dtm_max_x": max_x,
        "dtm_min_y": min_y,
        "dtm_max_y": max_y,
        "dtm_resolution": dtm_resolution,
        "dtm_shape": list(grid_z.shape),
        "dtm_requested_method": method,
        "dtm_effective_method": effective_method,
        "dtm_ground_unique_xy_count": int(unique_xy_count),
    }


def normalize_olive_height(olive_points, dtm_interpolator):
    ground_z = dtm_interpolator(olive_points[:, :2])
    invalid = np.isnan(ground_z)

    if invalid.any():
        valid_ground = ground_z[~invalid]
        if valid_ground.size == 0:
            raise ValueError("All interpolated ground heights are NaN")
        ground_z[invalid] = np.median(valid_ground)

    normalized_z = olive_points[:, 2] - ground_z
    normalized_z[normalized_z < 0] = 0

    normalized_points = olive_points.copy()
    normalized_points[:, 2] = normalized_z

    return normalized_points, ground_z


def build_grid(x, y, resolution):
    min_x = float(np.min(x))
    min_y = float(np.min(y))
    max_x = float(np.max(x))
    max_y = float(np.max(y))

    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_y - min_y) / resolution)) + 1

    col = np.floor((x - min_x) / resolution).astype(np.int64)
    row = np.floor((max_y - y) / resolution).astype(np.int64)

    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)

    return row[valid], col[valid], valid, min_x, min_y, max_x, max_y, height, width


def make_chm(row, col, z, height, width):
    chm = np.zeros((height, width), dtype=np.float32)
    np.maximum.at(chm, (row, col), z.astype(np.float32))
    return chm


def make_vertical_density(row, col, z, height, width, voxel_size):
    max_z = float(np.max(z))
    total_bins = max(1, int(np.ceil(max_z / voxel_size)))
    z_bin = np.floor(z / voxel_size).astype(np.int64)
    z_bin = np.clip(z_bin, 0, total_bins - 1)

    linear = (row * width + col) * total_bins + z_bin
    occupied = np.unique(linear)

    pixel_index = occupied // total_bins
    counts = np.bincount(pixel_index, minlength=height * width)

    return counts.reshape(height, width).astype(np.float32) / float(total_bins)


def make_gradient_chm(chm):
    gy, gx = np.gradient(chm.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def make_lbp_chm(chm):
    center = chm.astype(np.float32)
    padded = np.pad(center, 1, mode="edge")

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp.astype(np.float32) / 255.0


def normalize_band(band):
    band = band.astype(np.float32)
    min_value = float(np.min(band))
    max_value = float(np.max(band))
    if max_value <= min_value:
        return np.zeros_like(band, dtype=np.float32)
    return (band - min_value) / (max_value - min_value)


def save_preview_png(array, output_path):
    try:
        from PIL import Image
    except ImportError:
        return False

    image = (normalize_band(array) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)
    return True


def should_skip_existing(output_dir, stem):
    sample_dir = output_dir / stem
    metadata_path = sample_dir / f"{stem}_metadata.json"
    feature_path = sample_dir / f"{stem}_4ch.npy"
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError:
        return False
    status = metadata.get("status")
    if status == "ok":
        return feature_path.exists()
    return status in {"no_olive_points", "degenerate_grid"}


def process_file(txt_path, output_dir, semantic_id, resolution, voxel_size, dtm_method, dtm_resolution, csf_cloth_resolution, lower_cell_size, lower_percentile, griddata_method, column_spec=None, has_header=False):
    data = read_point_cloud(txt_path, column_spec=column_spec, has_header=has_header)

    with open(txt_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
    if has_header:
        available_columns = _read_header(first_line)
    else:
        if "," in first_line:
            available_columns = len(first_line.strip().split(","))
        else:
            available_columns = len(first_line.strip().split())

    spec = column_spec or "x,y,z,semantic_id"
    selected = _parse_column_spec(spec, available_columns)
    role_names = ["x", "y", "z", "semantic_id"]
    if not has_header:
        name_list = list(role_names)
    else:
        name_list = [available_columns[i] for i in selected]

    role_aliases = {
        "x": ["x"],
        "y": ["y"],
        "z": ["z"],
        "semantic_id": ["semantic_id", "semantic", "class", "label", "category"],
    }

    def _name_to_index(target_name):
        candidates = role_aliases.get(target_name, [target_name])
        for cand in candidates:
            for name in (cand, cand.lower(), cand.upper()):
                if name in name_list:
                    return name_list.index(name)
        for i, name in enumerate(name_list):
            if name.lower() in [c.lower() for c in candidates]:
                return i
        raise ValueError(
            f"Column for '{target_name}' not found. Selected columns: {name_list}\n"
            f"--columns 必须按 x,y,z,semantic_id 的顺序给出 (列名或列号均可)。"
        )

    col_x = _name_to_index("x")
    col_y = _name_to_index("y")
    col_z = _name_to_index("z")
    col_sem = _name_to_index("semantic_id")

    all_xyz = data[:, [col_x, col_y, col_z]].astype(np.float64)
    olive = data[data[:, col_sem].astype(np.int64) == semantic_id].astype(np.float64)

    if olive.shape[0] == 0:
        sample_dir = output_dir / txt_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": str(txt_path),
            "status": "no_olive_points",
            "semantic_id": semantic_id,
            "all_point_count": int(data.shape[0]),
            "olive_point_count": 0,
            "column_spec": spec,
            "selected_columns": name_list,
        }
        with open(sample_dir / f"{txt_path.stem}_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"[SKIP] {txt_path.name}: no olive points with semantic_id={semantic_id}")
        return

    if dtm_method == "csf":
        ground_points = split_ground_by_csf(all_xyz, csf_cloth_resolution)
    elif dtm_method == "lower_envelope":
        ground_points = split_ground_by_lower_envelope(all_xyz, lower_cell_size, lower_percentile)
    else:
        raise ValueError(f"Unsupported dtm_method: {dtm_method}")

    dtm_interpolator, dtm_meta = build_dtm_interpolator(ground_points, dtm_resolution, griddata_method)
    olive_normalized, olive_ground_z = normalize_olive_height(olive[:, [col_x, col_y, col_z]], dtm_interpolator)

    x = olive_normalized[:, 0]
    y = olive_normalized[:, 1]
    z = olive_normalized[:, 2]

    row, col, valid, min_x, min_y, max_x, max_y, height, width = build_grid(x, y, resolution)
    z = z[valid]

    if height < 2 or width < 2:
        sample_dir = output_dir / txt_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in sample_dir.glob(f"{txt_path.stem}_*"):
            if stale_path.name != f"{txt_path.stem}_metadata.json" and stale_path.is_file():
                stale_path.unlink()
        metadata = {
            "source": str(txt_path),
            "status": "degenerate_grid",
            "semantic_id": semantic_id,
            "all_point_count": int(data.shape[0]),
            "olive_point_count": int(olive.shape[0]),
            "resolution": resolution,
            "height": height,
            "width": width,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "column_spec": spec,
            "selected_columns": name_list,
        }
        with open(sample_dir / f"{txt_path.stem}_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"[SKIP] {txt_path.name}: degenerate grid height={height}, width={width}")
        return

    chm = make_chm(row, col, z, height, width)
    vd = make_vertical_density(row, col, z, height, width, voxel_size)
    g_chm = make_gradient_chm(chm)
    lbp_chm = make_lbp_chm(chm)

    feature_4ch = np.stack([vd, chm, g_chm, lbp_chm], axis=0).astype(np.float32)

    sample_dir = output_dir / txt_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(sample_dir / f"{txt_path.stem}_4ch.npy", feature_4ch)
    np.savez_compressed(
        sample_dir / f"{txt_path.stem}_features.npz",
        VD=vd,
        CHM=chm,
        G_CHM=g_chm,
        LBP_CHM=lbp_chm,
        feature_4ch=feature_4ch,
    )

    for name, band in zip(CHANNEL_NAMES, feature_4ch):
        np.save(sample_dir / f"{txt_path.stem}_{name}.npy", band)
        save_preview_png(band, sample_dir / f"{txt_path.stem}_{name}.png")

    metadata = {
        "source": str(txt_path),
        "status": "ok",
        "semantic_id": semantic_id,
        "all_point_count": int(data.shape[0]),
        "olive_point_count": int(olive.shape[0]),
        "ground_point_count": int(ground_points.shape[0]),
        "resolution": resolution,
        "voxel_size": voxel_size,
        "dtm_method": dtm_method,
        "csf_cloth_resolution": csf_cloth_resolution,
        "lower_cell_size": lower_cell_size,
        "lower_percentile": lower_percentile,
        "griddata_method": griddata_method,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "height": height,
        "width": width,
        "olive_height_min": float(np.min(z)),
        "olive_height_max": float(np.max(z)),
        "olive_height_mean": float(np.mean(z)),
        "olive_ground_z_min": float(np.min(olive_ground_z)),
        "olive_ground_z_max": float(np.max(olive_ground_z)),
        "channel_order": CHANNEL_NAMES,
        "feature_4ch_shape": list(feature_4ch.shape),
        "column_spec": spec,
        "selected_columns": name_list,
        **dtm_meta,
    }

    with open(sample_dir / f"{txt_path.stem}_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"[OK] {txt_path.name}: {feature_4ch.shape} -> {sample_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="从点云 .txt 生成橄榄树冠 4 通道特征栅格 (VD/CHM/G_CHM/LBP_CHM)",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/test"),
                        help="输入点云目录，里面放若干 *.txt 文件，每行一个点")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_3_new/test_4ch_features"),
                        help="输出根目录，每个输入文件会生成同名子文件夹")
    parser.add_argument("--semantic-id", type=int, default=1,
                        help="语义类别 ID，表示橄榄树点 (默认 1)")
    parser.add_argument("--columns", type=str, default="x,y,z,semantic_id",
                        help="需要读取的列，逗号分隔，支持列名 (如 x) 或列号 (如 0,1,2)。4 通道特征只需 x,y,z,semantic_id")
    parser.add_argument("--no-header", dest="has_header", action="store_false",
                        help="声明输入文件没有表头行 (默认)")
    parser.add_argument("--has-header", dest="has_header", action="store_true",
                        help="声明输入文件首行是表头行 (例如: x,y,z,class)")
    parser.set_defaults(has_header=False)
    parser.add_argument("--resolution", type=float, default=0.10,
                        help="输出栅格分辨率 (米/像素)，越小越细，默认 0.10")
    parser.add_argument("--voxel-size", type=float, default=0.25,
                        help="垂直密度统计的体素高度 (米)，默认 0.25")
    parser.add_argument("--dtm-method", choices=["csf", "lower_envelope"], default="csf",
                        help="DTM 地面估计方法: csf=布料模拟滤波, lower_envelope=低包络，默认 csf")
    parser.add_argument("--dtm-resolution", type=float, default=0.5,
                        help="DTM 插值格网分辨率 (米)，默认 0.5")
    parser.add_argument("--csf-cloth-resolution", type=float, default=0.5,
                        help="CSF 布料网格分辨率 (米)，越小越细，默认 0.5")
    parser.add_argument("--lower-cell-size", type=float, default=1.0,
                        help="lower_envelope 方法的格网边长 (米)，默认 1.0")
    parser.add_argument("--lower-percentile", type=float, default=5.0,
                        help="lower_envelope 方法中每格取地面点的 Z 百分位 (0-100)，默认 5")
    parser.add_argument("--griddata-method", choices=["linear", "nearest", "cubic"], default="linear",
                        help="DTM 插值算法，默认 linear")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {args.input_dir}")

    print(f"[INFO] input       = {args.input_dir}")
    print(f"[INFO] output      = {args.output_dir}")
    print(f"[INFO] columns     = {args.columns}  has_header={args.has_header}  dtm={args.dtm_method}")

    for txt_path in txt_files:
        if should_skip_existing(args.output_dir, txt_path.stem):
            print(f"[EXISTS] {txt_path.name}: completed or skipped metadata found")
            continue
        process_file(
            txt_path=txt_path,
            output_dir=args.output_dir,
            semantic_id=args.semantic_id,
            resolution=args.resolution,
            voxel_size=args.voxel_size,
            dtm_method=args.dtm_method,
            dtm_resolution=args.dtm_resolution,
            csf_cloth_resolution=args.csf_cloth_resolution,
            lower_cell_size=args.lower_cell_size,
            lower_percentile=args.lower_percentile,
            griddata_method=args.griddata_method,
            column_spec=args.columns,
            has_header=args.has_header,
        )


if __name__ == "__main__":
    main()