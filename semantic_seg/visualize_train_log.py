import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRAIN_PATTERN = re.compile(
    r"Train (?P<epoch>\d+), loss: (?P<loss>[\d.]+), "
    r"train acc: (?P<acc>[\d.]+), train ins_iou: (?P<ins_iou>[\d.]+)"
)
TEST_PATTERN = re.compile(
    r"Test (?P<epoch>\d+), loss: (?P<loss>[\d.]+), "
    r"test acc: (?P<acc>[\d.]+)\s+test ins_iou: (?P<ins_iou>[\d.]+)"
)
LR_PATTERN = re.compile(r"Learning rate: (?P<lr>[\d.eE+-]+)")


def parse_log(log_path):
    train_rows = []
    test_rows = []
    pending_lr = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lr_match = LR_PATTERN.search(line)
            if lr_match:
                pending_lr = float(lr_match.group("lr"))
                continue

            train_match = TRAIN_PATTERN.search(line)
            if train_match:
                train_rows.append({
                    "epoch": int(train_match.group("epoch")),
                    "loss": float(train_match.group("loss")),
                    "acc": float(train_match.group("acc")),
                    "ins_iou": float(train_match.group("ins_iou")),
                    "lr": pending_lr,
                })
                continue

            test_match = TEST_PATTERN.search(line)
            if test_match:
                test_rows.append({
                    "epoch": int(test_match.group("epoch")),
                    "loss": float(test_match.group("loss")),
                    "acc": float(test_match.group("acc")),
                    "ins_iou": float(test_match.group("ins_iou")),
                })

    train_df = pd.DataFrame(train_rows).drop_duplicates(subset=["epoch"], keep="last")
    test_df = pd.DataFrame(test_rows).drop_duplicates(subset=["epoch"], keep="last")

    if train_df.empty:
        raise ValueError(f"No PointMLP train records found in {log_path}")
    if test_df.empty:
        raise ValueError(f"No PointMLP test records found in {log_path}")

    merged = train_df.merge(test_df, on="epoch", how="outer", suffixes=("_train", "_test")).sort_values("epoch")
    return train_df, test_df, merged


def plot_curves(train_df, test_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    axes[0, 0].plot(train_df["epoch"].values, train_df["loss"].values, color="#1f77b4", label="train loss")
    axes[0, 0].plot(test_df["epoch"].values, test_df["loss"].values, color="#ff7f0e", label="test loss")
    best_loss_row = test_df.loc[test_df["loss"].idxmin()]
    axes[0, 0].scatter(best_loss_row["epoch"], best_loss_row["loss"], color="red", zorder=3, label="best test loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_df["epoch"].values, train_df["acc"].values, color="#2ca02c", label="train acc")
    axes[0, 1].plot(test_df["epoch"].values, test_df["acc"].values, color="#17becf", label="test acc")
    best_acc_row = test_df.loc[test_df["acc"].idxmax()]
    axes[0, 1].scatter(best_acc_row["epoch"], best_acc_row["acc"], color="red", zorder=3, label="best test acc")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Acc")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(train_df["epoch"].values, train_df["ins_iou"].values, color="#9467bd", label="train ins_iou")
    axes[1, 0].plot(test_df["epoch"].values, test_df["ins_iou"].values, color="#d62728", label="test ins_iou")
    best_iou_row = test_df.loc[test_df["ins_iou"].idxmax()]
    axes[1, 0].scatter(best_iou_row["epoch"], best_iou_row["ins_iou"], color="red", zorder=3, label="best test ins_iou")
    axes[1, 0].set_title("Instance/Sample mIoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    if "lr" in train_df.columns and train_df["lr"].notna().any():
        axes[1, 1].plot(train_df["epoch"].values, train_df["lr"].values, color="#8c564b", label="lr")
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_best_checkpoints(test_df, work_dir):
    best_iou = test_df.loc[test_df["ins_iou"].idxmax()]
    best_acc = test_df.loc[test_df["acc"].idxmax()]
    best_loss = test_df.loc[test_df["loss"].idxmin()]

    print("\nBest test ins_iou:")
    print(f"  epoch: {int(best_iou['epoch'])}")
    print(f"  ins_iou: {best_iou['ins_iou']:.6f}")
    print(f"  checkpoint: {work_dir / 'best_insiou_model.pth'}")
    print(f"  checkpoint exists: {(work_dir / 'best_insiou_model.pth').exists()}")

    print("\nBest test acc:")
    print(f"  epoch: {int(best_acc['epoch'])}")
    print(f"  acc: {best_acc['acc']:.6f}")
    print(f"  checkpoint: {work_dir / 'best_acc_model.pth'}")
    print(f"  checkpoint exists: {(work_dir / 'best_acc_model.pth').exists()}")

    print("\nLowest test loss:")
    print(f"  epoch: {int(best_loss['epoch'])}")
    print(f"  loss: {best_loss['loss']:.6f}")


def main():
    parser = argparse.ArgumentParser("plot_pointmlp_partseg_training_log")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/workspace/pointMLP-pytorch-main/part_segmentation/checkpoints/pointMLP_olive_olive",
        help="Directory containing PointMLP part segmentation training log and checkpoints.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Specific log file path. If omitted, the newest .log file in work-dir is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save csv and png outputs. Default is work-dir/training_plots.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    if args.log_file is None:
        log_files = sorted(work_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            raise FileNotFoundError(f"No .log files found in {work_dir}")
        log_path = log_files[0]
    else:
        log_path = Path(args.log_file)

    output_dir = Path(args.output_dir) if args.output_dir else work_dir / "training_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, merged = parse_log(log_path)

    train_csv = output_dir / "pointmlp_train_metrics.csv"
    test_csv = output_dir / "pointmlp_test_metrics.csv"
    merged_csv = output_dir / "pointmlp_metrics_merged.csv"
    plot_path = output_dir / "pointmlp_training_curves.png"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    merged.to_csv(merged_csv, index=False)

    plot_curves(train_df, test_df, plot_path)

    print(f"Parsed log: {log_path}")
    print(f"Saved train metrics: {train_csv}")
    print(f"Saved test metrics: {test_csv}")
    print(f"Saved merged metrics: {merged_csv}")
    print(f"Saved plot: {plot_path}")

    print_best_checkpoints(test_df, work_dir)


if __name__ == "__main__":
    main()