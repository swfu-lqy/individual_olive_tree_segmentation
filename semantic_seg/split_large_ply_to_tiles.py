import argparse
import math
import os
from pathlib import Path

import numpy as np


PLY_TYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_ply_header(path):
    header = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY file")
            text = line.decode("utf-8", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                header_bytes = f.tell()
                break

    ply_format = None
    vertex_count = None
    properties = []
    in_vertex = False

    for line in header:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "format":
            ply_format = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            if parts[1] == "list":
                raise ValueError("List properties in vertex are not supported")
            properties.append((parts[1], parts[-1]))

    if ply_format is None:
        raise ValueError("PLY format not found")
    if vertex_count is None:
        raise ValueError("PLY vertex element not found")

    prop_names = [name for _, name in properties]
    for name in ["x", "y", "z"]:
        if name not in prop_names:
            raise ValueError("PLY must contain x y z")

    if all(name in prop_names for name in ["red", "green", "blue"]):
        color_names = ("red", "green", "blue")
    elif all(name in prop_names for name in ["r", "g", "b"]):
        color_names = ("r", "g", "b")
    else:
        raise ValueError("PLY must contain red green blue or r g b")

    return ply_format, vertex_count, properties, prop_names, color_names, header_bytes, len(header)


def build_binary_memmap(path, ply_format, vertex_count, properties, header_bytes):
    endian = "<" if ply_format == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + PLY_TYPE_MAP[prop_type]) for prop_type, name in properties])
    return np.memmap(path, dtype=dtype, mode="r", offset=header_bytes, shape=(vertex_count,))


def get_binary_bounds(vertices):
    x_min = float(np.min(vertices["x"]))
    x_max = float(np.max(vertices["x"]))
    y_min = float(np.min(vertices["y"]))
    y_max = float(np.max(vertices["y"]))
    return x_min, x_max, y_min, y_max


def update_bounds_from_points(points, bounds):
    x_min, x_max, y_min, y_max = bounds
    x_min = min(x_min, float(np.min(points[:, 0])))
    x_max = max(x_max, float(np.max(points[:, 0])))
    y_min = min(y_min, float(np.min(points[:, 1])))
    y_max = max(y_max, float(np.max(points[:, 1])))
    return x_min, x_max, y_min, y_max


def get_txt_bounds(path, chunk_lines, skip_rows=0):
    bounds = (float("inf"), -float("inf"), float("inf"), -float("inf"))
    processed = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(skip_rows):
            next(f, None)
        chunk = []
        for line in f:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= chunk_lines:
                arr = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                bounds = update_bounds_from_points(arr, bounds)
                processed += arr.shape[0]
                print(f"bounds processed {processed}")
                chunk = []
        if chunk:
            arr = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            bounds = update_bounds_from_points(arr, bounds)
            processed += arr.shape[0]
            print(f"bounds processed {processed}")
    return bounds


def get_ascii_bounds(path, header_lines, chunk_lines):
    x_min, x_max = float("inf"), -float("inf")
    y_min, y_max = float("inf"), -float("inf")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(header_lines):
            next(f)
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_lines:
                arr = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                x_min, x_max, y_min, y_max = update_bounds_from_points(arr, (x_min, x_max, y_min, y_max))
                chunk = []
        if chunk:
            arr = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            x_min, x_max, y_min, y_max = update_bounds_from_points(arr, (x_min, x_max, y_min, y_max))
    return x_min, x_max, y_min, y_max


def make_windows(x_min, x_max, y_min, y_max, block_size, stride):
    xs = []
    ys = []

    x = x_min
    while x <= x_max:
        xs.append(x)
        x += stride
    if xs[-1] + block_size < x_max:
        xs.append(x_max - block_size)

    y = y_max - block_size
    while y >= y_min:
        ys.append(y)
        y -= stride
    if ys[-1] > y_min:
        ys.append(y_min)

    windows = []
    for row_idx, y0 in enumerate(ys):
        for col_idx, x0 in enumerate(xs):
            x1 = x0 + block_size
            y1 = y0 + block_size
            windows.append((row_idx, col_idx, x0, x1, y0, y1))
    return windows


def ensure_tile_files(output_dir, windows):
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for row_idx, col_idx, _, _, _, _ in windows:
        tile_paths.append(output_dir / f"tile_r{row_idx:04d}_c{col_idx:04d}.txt")
    return tile_paths


def points_to_array_from_binary(vertices, prop_names, color_names, start, end):
    chunk = vertices[start:end]
    xyz = np.vstack([chunk["x"], chunk["y"], chunk["z"]]).T.astype(np.float64)
    rgb = np.vstack([chunk[color_names[0]], chunk[color_names[1]], chunk[color_names[2]]]).T.astype(np.float64)
    if rgb.max() <= 1.0:
        rgb *= 255.0
    return np.concatenate([xyz, rgb], axis=1)


def write_points_to_windows(points, windows, tile_paths):
    x = points[:, 0]
    y = points[:, 1]
    for window, tile_path in zip(windows, tile_paths):
        _, _, x0, x1, y0, y1 = window
        mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if np.any(mask):
            with open(tile_path, "ab") as f:
                np.savetxt(f, points[mask], fmt="%.6f %.6f %.6f %.0f %.0f %.0f")


def split_binary_ply(path, vertices, prop_names, color_names, windows, tile_paths, chunk_size):
    total = vertices.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        points = points_to_array_from_binary(vertices, prop_names, color_names, start, end)
        write_points_to_windows(points, windows, tile_paths)
        print(f"processed {end}/{total}")


def split_txt_points(path, windows, tile_paths, chunk_lines, skip_rows=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(skip_rows):
            next(f, None)
        chunk = []
        processed = 0
        for line in f:
            if not line.strip():
                continue
            chunk.append(line)
            if len(chunk) >= chunk_lines:
                points = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                write_points_to_windows(points, windows, tile_paths)
                processed += points.shape[0]
                print(f"processed {processed}")
                chunk = []
        if chunk:
            points = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
            if points.ndim == 1:
                points = points.reshape(1, -1)
            write_points_to_windows(points, windows, tile_paths)
            processed += points.shape[0]
            print(f"processed {processed}")


def split_ascii_ply(path, header_lines, windows, tile_paths, chunk_lines):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(header_lines):
            next(f)
        chunk = []
        processed = 0
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_lines:
                points = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                write_points_to_windows(points, windows, tile_paths)
                processed += len(chunk)
                print(f"processed {processed}")
                chunk = []
        if chunk:
            points = np.loadtxt(chunk, dtype=np.float64, usecols=(0, 1, 2, 3, 4, 5))
            if points.ndim == 1:
                points = points.reshape(1, -1)
            write_points_to_windows(points, windows, tile_paths)
            processed += len(chunk)
            print(f"processed {processed}")


def remove_empty_tiles(tile_paths):
    removed = 0
    for path in tile_paths:
        if not path.exists() or path.stat().st_size == 0:
            if path.exists():
                path.unlink()
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser("split_large_pointcloud_to_overlapped_txt_tiles")
    parser.add_argument("--input", type=str, default="/workspace/pointMLP-pytorch-main/part_segmentation/data/olive_data/inference_entire_block/output.ply")
    parser.add_argument("--output-dir", type=str, default="/workspace/pointMLP-pytorch-main/part_segmentation/data/olive_data/inference_entire_block_parting")
    parser.add_argument("--block-size", type=float, default=35.0)
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    parser.add_argument("--skip-rows", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    stride = args.block_size * (1.0 - args.overlap)

    if input_path.suffix.lower() == ".txt":
        ply_format = "txt"
        vertex_count = "unknown"
        vertices = None
        prop_names = None
        color_names = None
        header_lines = args.skip_rows
        x_min, x_max, y_min, y_max = get_txt_bounds(input_path, args.chunk_size, args.skip_rows)
    else:
        ply_format, vertex_count, properties, prop_names, color_names, header_bytes, header_lines = parse_ply_header(input_path)
        if ply_format == "ascii":
            x_min, x_max, y_min, y_max = get_ascii_bounds(input_path, header_lines, args.chunk_size)
            vertices = None
        elif ply_format in ["binary_little_endian", "binary_big_endian"]:
            vertices = build_binary_memmap(input_path, ply_format, vertex_count, properties, header_bytes)
            x_min, x_max, y_min, y_max = get_binary_bounds(vertices)
        else:
            raise ValueError(f"Unsupported PLY format: {ply_format}")

    print(f"format: {ply_format}")
    print(f"points: {vertex_count}")
    print(f"x range: {x_min:.3f} - {x_max:.3f}")
    print(f"y range: {y_min:.3f} - {y_max:.3f}")
    print(f"block_size: {args.block_size}")
    print(f"stride: {stride}")

    windows = make_windows(x_min, x_max, y_min, y_max, args.block_size, stride)
    tile_paths = ensure_tile_files(output_dir, windows)

    print(f"tiles: {len(windows)}")
    print(f"output: {output_dir}")

    if input_path.suffix.lower() == ".txt":
        split_txt_points(input_path, windows, tile_paths, args.chunk_size, args.skip_rows)
    elif vertices is not None:
        split_binary_ply(input_path, vertices, prop_names, color_names, windows, tile_paths, args.chunk_size)
    else:
        split_ascii_ply(input_path, header_lines, windows, tile_paths, args.chunk_size)

    removed = remove_empty_tiles(tile_paths)
    print(f"done, removed empty tiles: {removed}")


if __name__ == "__main__":
    main()