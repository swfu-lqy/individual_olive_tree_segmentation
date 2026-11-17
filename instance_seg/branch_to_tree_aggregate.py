import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import Birch


def read_point_cloud(txt_path):
    try:
        cloud = np.loadtxt(txt_path)
    except ValueError:
        cloud = np.loadtxt(txt_path, delimiter=",")

    if cloud.ndim == 1:
        cloud = cloud.reshape(1, -1)
    if cloud.shape[1] < 9:
        raise ValueError(f"{txt_path} should have at least 9 columns: x y z r g b gt_semantic gt_instance pred_semantic")

    return cloud[:, :9].astype(np.float64)


def read_world_boxes(box_path, conf_threshold):
    if not box_path.exists():
        return []

    boxes = []
    with open(box_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split()
        expected = ["pred_instance_id", "cls_id", "confidence", "x_min", "y_min", "x_max", "y_max", "center_x", "center_y"]
        if header != expected:
            raise ValueError(f"{box_path} header should be: {' '.join(expected)}")

        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            confidence = float(parts[2])
            if confidence < conf_threshold:
                continue
            boxes.append({
                "pred_instance_id": int(float(parts[0])),
                "cls_id": int(float(parts[1])),
                "confidence": confidence,
                "x_min": float(parts[3]),
                "y_min": float(parts[4]),
                "x_max": float(parts[5]),
                "y_max": float(parts[6]),
                "center_x": float(parts[7]),
                "center_y": float(parts[8]),
            })

    boxes.sort(key=lambda item: item["pred_instance_id"])
    return boxes


def cluster_outside_subclusters(sub_centroids_xy, outside_indices, start_instance_id, cluster_distance):
    if not outside_indices:
        return {}, 0

    outside_xy = sub_centroids_xy[outside_indices]
    tree = cKDTree(outside_xy)
    pairs = tree.query_pairs(r=cluster_distance)
    parent = np.arange(len(outside_indices), dtype=np.int32)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i, j in pairs:
        union(i, j)

    root_to_instance = {}
    outside_assignments = {}
    next_instance_id = start_instance_id
    for local_i, global_i in enumerate(outside_indices):
        root = find(local_i)
        if root not in root_to_instance:
            root_to_instance[root] = next_instance_id
            next_instance_id += 1
        outside_assignments[global_i] = root_to_instance[root]

    return outside_assignments, len(root_to_instance)


def build_box_inside_mask(points_xy, world_boxes):
    if not world_boxes:
        return np.zeros(points_xy.shape[0], dtype=bool)

    box_bounds = np.asarray(
        [[box["x_min"], box["x_max"], box["y_min"], box["y_max"]] for box in world_boxes],
        dtype=np.float64,
    )
    inside_mask = np.zeros(points_xy.shape[0], dtype=bool)
    for x_min, x_max, y_min, y_max in box_bounds:
        inside_mask |= (
            (points_xy[:, 0] >= x_min)
            & (points_xy[:, 0] <= x_max)
            & (points_xy[:, 1] >= y_min)
            & (points_xy[:, 1] <= y_max)
        )
    return inside_mask


def build_centers_xyz(world_boxes, foreground_xyz, cylinder_radius, cylinder_z_mode):
    centers_xy = np.asarray([[box["center_x"], box["center_y"]] for box in world_boxes], dtype=np.float64)
    centers_z = np.empty(len(world_boxes), dtype=np.float64)
    fallback_z = float(np.mean(foreground_xyz[:, 2]))

    for i, center_xy in enumerate(centers_xy):
        distances_xy = np.sqrt((foreground_xyz[:, 0] - center_xy[0]) ** 2 + (foreground_xyz[:, 1] - center_xy[1]) ** 2)
        cylinder_z = foreground_xyz[distances_xy <= cylinder_radius, 2]
        if cylinder_z.size == 0:
            centers_z[i] = fallback_z
        elif cylinder_z_mode == "min":
            centers_z[i] = float(np.min(cylinder_z))
        elif cylinder_z_mode == "mid":
            centers_z[i] = float((np.min(cylinder_z) + np.max(cylinder_z)) / 2.0)
        elif cylinder_z_mode == "max":
            centers_z[i] = float(np.max(cylinder_z))
        else:
            raise ValueError(f"Unsupported cylinder_z_mode: {cylinder_z_mode}")

    return np.column_stack([centers_xy, centers_z])


def assign_subclusters_to_boxes(sub_centroids, world_boxes, centers_xyz, instance_ids, use_box_constraint, outside_box_policy, outside_cluster_distance):
    tree = cKDTree(centers_xyz)
    _, nearest_center = tree.query(sub_centroids)

    if not use_box_constraint:
        return instance_ids[nearest_center].astype(np.int32), 0, 0, 0

    box_bounds = np.asarray(
        [[box["x_min"], box["x_max"], box["y_min"], box["y_max"]] for box in world_boxes],
        dtype=np.float64,
    )
    assigned_instances = np.full(sub_centroids.shape[0], -1, dtype=np.int32)
    fallback_count = 0
    ignored_count = 0
    new_instance_count = 0
    outside_new_indices = []

    for i, centroid in enumerate(sub_centroids[:, :2]):
        x, y = centroid
        inside = (
            (x >= box_bounds[:, 0])
            & (x <= box_bounds[:, 1])
            & (y >= box_bounds[:, 2])
            & (y <= box_bounds[:, 3])
        )
        candidate_indices = np.where(inside)[0]
        if candidate_indices.size == 0:
            if outside_box_policy == "nearest":
                assigned_instances[i] = int(instance_ids[nearest_center[i]])
                fallback_count += 1
            elif outside_box_policy == "ignore":
                assigned_instances[i] = -1
                ignored_count += 1
            elif outside_box_policy == "new_instance":
                outside_new_indices.append(i)
            else:
                raise ValueError(f"Unsupported outside_box_policy: {outside_box_policy}")
            continue
        centers = centers_xyz[candidate_indices]
        distances = np.sqrt(
            (centers[:, 0] - sub_centroids[i, 0]) ** 2
            + (centers[:, 1] - sub_centroids[i, 1]) ** 2
            + (centers[:, 2] - sub_centroids[i, 2]) ** 2
        )
        assigned_instances[i] = int(instance_ids[candidate_indices[int(np.argmin(distances))]])

    if outside_box_policy == "new_instance" and outside_new_indices:
        start_instance_id = int(np.max(instance_ids)) + 1 if instance_ids.size > 0 else 0
        outside_assignments, new_instance_count = cluster_outside_subclusters(
            sub_centroids_xy=sub_centroids[:, :2],
            outside_indices=outside_new_indices,
            start_instance_id=start_instance_id,
            cluster_distance=outside_cluster_distance,
        )
        for sub_index, instance_id in outside_assignments.items():
            assigned_instances[sub_index] = instance_id

    return assigned_instances, fallback_count, ignored_count, new_instance_count


def two_stage_center_guided_clustering(cloud, world_boxes, output_txt, semantic_id=1, birch_factor=10, threshold=0.5, use_box_constraint=False, outside_box_policy="nearest", outside_cluster_distance=1.5, cylinder_radius=0.5, cylinder_z_mode="mid"):
    pred_instance = np.full(cloud.shape[0], -1, dtype=np.int32)
    foreground_mask = cloud[:, 8].astype(np.int64) == semantic_id
    foreground = cloud[foreground_mask]

    if foreground.shape[0] == 0:
        output = np.column_stack([cloud, pred_instance])
        save_output(output, output_txt)
        print(f"[EMPTY] no pred_semantic={semantic_id} points -> {output_txt}")
        return output

    if not world_boxes:
        output = np.column_stack([cloud, pred_instance])
        save_output(output, output_txt)
        print(f"[EMPTY] no world boxes -> {output_txt}")
        return output

    xyz = foreground[:, :3]
    active_foreground_mask = foreground_mask.copy()
    if use_box_constraint and outside_box_policy == "ignore":
        inside_box_mask = build_box_inside_mask(xyz[:, :2], world_boxes)
        active_foreground_indices = np.where(foreground_mask)[0][inside_box_mask]
        active_foreground_mask = np.zeros(cloud.shape[0], dtype=bool)
        active_foreground_mask[active_foreground_indices] = True
        foreground = cloud[active_foreground_mask]
        xyz = foreground[:, :3]
        if foreground.shape[0] == 0:
            output = np.column_stack([cloud, pred_instance])
            save_output(output, output_txt)
            print(f"[EMPTY] no pred_semantic={semantic_id} points inside YOLO boxes -> {output_txt}")
            return output

    centers_xyz = build_centers_xyz(world_boxes, xyz, cylinder_radius, cylinder_z_mode)
    instance_ids = np.asarray([box["pred_instance_id"] for box in world_boxes], dtype=np.int32)
    k = len(centers_xyz)
    n_subclusters = max(k, k * birch_factor)

    print("=" * 60)
    print("YOLO预测中心数量:", k)
    print("前景 olive 点数:", int(np.sum(foreground_mask)))
    if use_box_constraint and outside_box_policy == "ignore":
        print("框内参与聚类 olive 点数:", foreground.shape[0])
        print("框外预过滤忽略 olive 点数:", int(np.sum(foreground_mask) - foreground.shape[0]))
    print("BIRCH目标小簇数:", n_subclusters)

    birch = Birch(
        threshold=threshold,
        branching_factor=50,
        n_clusters=n_subclusters,
    )
    sub_labels = birch.fit_predict(xyz)
    unique_sub = np.unique(sub_labels)
    print("实际生成小簇数量:", len(unique_sub))

    sub_centroids = np.asarray([xyz[sub_labels == label].mean(axis=0) for label in unique_sub], dtype=np.float64)
    sub_instances, fallback_count, ignored_count, new_instance_count = assign_subclusters_to_boxes(
        sub_centroids=sub_centroids,
        world_boxes=world_boxes,
        centers_xyz=centers_xyz,
        instance_ids=instance_ids,
        use_box_constraint=use_box_constraint,
        outside_box_policy=outside_box_policy,
        outside_cluster_distance=outside_cluster_distance,
    )
    subcluster_to_instance = {sub_id: int(instance_id) for sub_id, instance_id in zip(unique_sub, sub_instances)}
    fg_instance = np.asarray([subcluster_to_instance[sub] for sub in sub_labels], dtype=np.int32)
    print("YOLO框范围限制:", "开启" if use_box_constraint else "关闭")
    if use_box_constraint:
        print("框外小簇处理策略:", outside_box_policy)
        print("框外回退到最近中心的小簇数:", fallback_count)
        print("框外忽略为 -1 的小簇数:", ignored_count)
        print("框外新建实例组数:", new_instance_count)
        print("YOLO中心圆柱半径:", cylinder_radius)
        print("YOLO中心圆柱z参考位置:", cylinder_z_mode)
        if outside_box_policy == "new_instance":
            print("框外小簇质心聚类距离阈值:", outside_cluster_distance)
    pred_instance[active_foreground_mask] = fg_instance

    output = np.column_stack([cloud, pred_instance])
    save_output(output, output_txt)

    print("聚类完成！")
    print("输出文件:", output_txt)
    print("=" * 60)
    return output


def save_output(output, output_txt):
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    fmt = ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"]
    np.savetxt(output_txt, output, fmt=fmt)


def process_file(txt_path, box_dir, output_dir, semantic_id, conf_threshold, birch_factor, threshold, use_box_constraint, outside_box_policy, outside_cluster_distance, cylinder_radius, cylinder_z_mode):
    stem = txt_path.stem
    box_path = box_dir / f"{stem}_pred_boxes_world.txt"
    output_txt = output_dir / f"{stem}_BTA_instances.txt"

    cloud = read_point_cloud(txt_path)
    world_boxes = read_world_boxes(box_path, conf_threshold)
    output = two_stage_center_guided_clustering(
        cloud=cloud,
        world_boxes=world_boxes,
        output_txt=output_txt,
        semantic_id=semantic_id,
        birch_factor=birch_factor,
        threshold=threshold,
        use_box_constraint=use_box_constraint,
        outside_box_policy=outside_box_policy,
        outside_cluster_distance=outside_cluster_distance,
        cylinder_radius=cylinder_radius,
        cylinder_z_mode=cylinder_z_mode,
    )
    assigned_count = int(np.sum(output[:, -1] >= 0))
    print(f"[OK] {stem}: boxes={len(world_boxes)}, assigned_points={assigned_count} -> {output_txt}")


def main():
    parser = argparse.ArgumentParser(description="基于 YOLO 世界坐标框中心引导的两阶段 BIRCH 单木实例聚类")
    parser.add_argument("--point-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/olive_data_semantic_predicted"), help="输入点云 txt 目录，文件列至少包含 x,y,z,r,g,b,gt_semantic,gt_instance,pred_semantic")
    parser.add_argument("--box-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/yolo_maped_points_boxes"), help="YOLO 检测框已映射到点云坐标系后的目录，文件名需为 *_pred_boxes_world.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/Yolo_liqiying/olive_data_pointnet_predicted_result/two_stage_clustered_points"), help="输出聚类后的点云 txt 目录")
    parser.add_argument("--semantic-id", type=int, default=1, help="pred_semantic 列中代表 olive/tree 前景的语义类别 ID")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="只使用置信度大于等于该阈值的 YOLO 世界坐标框")
    parser.add_argument("--birch-factor", type=int, default=10, help="BIRCH 过分割倍率，目标小簇数量为 YOLO 框数量乘以该值")
    parser.add_argument("--threshold", type=float, default=0.5, help="BIRCH 聚类 threshold，数值越小小簇越细")
    parser.add_argument("--use-box-constraint", action="store_true", help="二阶段分配时启用 YOLO 框范围限制：小簇质心若落入一个或多个框内，则只在这些框内按中心距离选择实例")
    parser.add_argument("--outside-box-policy", choices=["nearest", "ignore", "new_instance"], default="nearest", help="启用 --use-box-constraint 后，小簇质心不在任何 YOLO 框内时的处理策略：nearest=回退到全局最近中心，ignore=标记为 -1，new_instance=将框外小簇质心按距离聚成若干新实例")
    parser.add_argument("--outside-cluster-distance", type=float, default=1.5, help="当 --outside-box-policy new_instance 时，框外小簇质心之间距离小于等于该阈值则归为同一个新实例，单位与点云坐标一致")
    parser.add_argument("--cylinder-radius", type=float, default=0.5, help="以 YOLO 世界坐标中心为圆心统计前景点 z 值的水平圆柱半径，单位与点云坐标一致")
    parser.add_argument("--cylinder-z-mode", choices=["min", "mid", "max"], default="mid", help="YOLO 中心三维化时采用圆柱内前景点的 z_min、z_mid 或 z_max 作为第三维")
    parser.add_argument("--pattern", type=str, default="*.txt", help="点云文件匹配模式")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(args.point_dir.glob(args.pattern))
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {args.point_dir}")

    print(f"[INFO] point_dir={args.point_dir}")
    print(f"[INFO] box_dir={args.box_dir}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] files={len(txt_files)}, semantic_id={args.semantic_id}, conf_threshold={args.conf_threshold}, use_box_constraint={args.use_box_constraint}, outside_box_policy={args.outside_box_policy}, outside_cluster_distance={args.outside_cluster_distance}")

    for txt_path in txt_files:
        process_file(
            txt_path=txt_path,
            box_dir=args.box_dir,
            output_dir=args.output_dir,
            semantic_id=args.semantic_id,
            conf_threshold=args.conf_threshold,
            birch_factor=args.birch_factor,
            threshold=args.threshold,
            use_box_constraint=args.use_box_constraint,
            outside_box_policy=args.outside_box_policy,
            outside_cluster_distance=args.outside_cluster_distance,
            cylinder_radius=args.cylinder_radius,
            cylinder_z_mode=args.cylinder_z_mode,
        )


if __name__ == "__main__":
    main()