from pathlib import Path
import argparse
import numpy as np


def load_txt(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def format_rows(data):
    lines = []
    for row in data:
        lines.append(
            f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
            f"{int(round(row[3]))} {int(round(row[4]))} {int(round(row[5]))} "
            f"{int(round(row[6]))} {int(round(row[7]))} {int(round(row[8]))}"
        )
    return "\n".join(lines) + "\n"


def merge_one_file(src_file, pred_file, out_file):
    src_data = load_txt(src_file)
    pred_data = load_txt(pred_file)

    if src_data.shape[1] < 6:
        raise ValueError(f"{src_file} has {src_data.shape[1]} columns, expected at least 6")

    if pred_data.shape[1] < 9:
        raise ValueError(f"{pred_file} has {pred_data.shape[1]} columns, expected at least 9")

    if src_data.shape[0] != pred_data.shape[0]:
        raise ValueError(
            f"Point count mismatch: {src_file.name} has {src_data.shape[0]} rows, "
            f"{pred_file.name} has {pred_data.shape[0]} rows"
        )

    merged = np.concatenate([src_data[:, :6], pred_data[:, -3:]], axis=1)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(format_rows(merged), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        default="/workspace/pointMLP-pytorch-main/part_segmentation/data/olive_data/inference_entire_block_parting",
    )
    parser.add_argument(
        "--pred-dir",
        default="/workspace/pointMLP-pytorch-main/part_segmentation/checkpoints/pointMLP_olive_olive_v2/inference_entire_block",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/pointMLP-pytorch-main/part_segmentation/checkpoints/pointMLP_olive_olive_v2/inference_entire_block_original_coord",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    src_files = sorted(src_dir.glob("*.txt"))
    if not src_files:
        raise FileNotFoundError(f"No txt files found in {src_dir}")

    merged_count = 0
    missing_count = 0
    skipped_count = 0

    for src_file in src_files:
        pred_file = pred_dir / f"{src_file.stem}_with_pred.txt"
        out_file = out_dir / f"{src_file.stem}_with_pred.txt"

        if not pred_file.exists():
            print(f"[missing] {pred_file}")
            missing_count += 1
            continue

        if out_file.exists() and not args.overwrite:
            print(f"[skip] {out_file}")
            skipped_count += 1
            continue

        merge_one_file(src_file, pred_file, out_file)
        print(f"[merged] {src_file.name} -> {out_file.name}")
        merged_count += 1

    print(
        f"Done. merged={merged_count}, missing={missing_count}, "
        f"skipped={skipped_count}, output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()