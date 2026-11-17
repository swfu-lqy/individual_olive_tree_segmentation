import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os

def get_all_files(root_dir):
    """获取所有文件的完整路径"""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    return all_files

def load_point_cloud(txt_path):
    """读取点云文件，假设列为: x y z r g b gt_sem gt_id pred_sem pred_id"""
    df = pd.read_csv(
        txt_path,
        sep=r"\s+|,",
        engine="python",
        header=None,
        names=["x", "y", "z", "r", "g", "b", "gt_sem", "gt_id", "pred_sem", "pred_id"],
#         names=["x", "y", "z", "r", "g", "b", "gt_sem", "gt_id", "pred_sem", "pred_id", "R", "G", "B",],
        # names=["x", "y", "z", "r", "g", "b", "gt_sem", "gt_id", "pred_sem", "pred_id", "confidence"],
    )
    return df

def build_instance_points(df, sem_col, id_col, target_sem=1):
    """
    构建实例点集
    返回: {instance_id: set(point_indices)}
    """
    instances = {}
    mask = df[sem_col] == target_sem
    
    for inst_id, group in df[mask].groupby(id_col):
        if inst_id < 0:
            continue
        instances[int(inst_id)] = set(group.index)
    
    return instances

def instance_iou(gt_points, pred_points):
    """计算两个实例的IoU"""
    inter = len(gt_points & pred_points)
    union = len(gt_points | pred_points)
    if union == 0:
        return 0.0
    return inter / union

def build_iou_matrix(gt_instances, pred_instances):
    """
    构建IoU矩阵
    返回: iou_matrix (N_gt, N_pred), gt_ids列表, pred_ids列表
    """
    gt_ids = list(gt_instances.keys())
    pred_ids = list(pred_instances.keys())
    
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    
    for i, gt_id in enumerate(gt_ids):
        gt_points = gt_instances[gt_id]
        for j, pred_id in enumerate(pred_ids):
            pred_points = pred_instances[pred_id]
            inter = len(gt_points & pred_points)
            union = len(gt_points | pred_points)
            if union > 0:
                iou_matrix[i, j] = inter / union
    
    return iou_matrix, gt_ids, pred_ids

def hungarian_match_instances(gt_instances, pred_instances, iou_threshold=0.5):
    """使用匈牙利算法进行实例匹配（最大IoU）"""
    if len(gt_instances) == 0 or len(pred_instances) == 0:
        return pd.DataFrame(), set(), set(), []
    
    # 构建IoU矩阵
    iou_matrix, gt_ids, pred_ids = build_iou_matrix(gt_instances, pred_instances)
    
    # 构建cost matrix（最小化）
    cost_matrix = 1.0 - iou_matrix
    
    # 匈牙利匹配
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_gt = set()
    matched_pred = set()
    matched_ious = []  # 新增：存储匹配的IoU值
    
    # 根据IoU阈值筛选有效匹配
    for i, j in zip(gt_idx, pred_idx):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            gt_id = gt_ids[i]
            pred_id = pred_ids[j]
            matches.append({
                "gt_id": gt_id,
                "pred_id": pred_id,
                "iou": iou
            })
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matched_ious.append(iou)  # 存储IoU值
    
    matches_df = pd.DataFrame(matches)
    return matches_df, matched_gt, matched_pred, matched_ious

def compute_semantic_iou(df, target_sem=1):
    """计算语义分割IoU"""
    gt_pos = df["gt_sem"] == target_sem
    pred_pos = df["pred_sem"] == target_sem
    
    tp = np.sum(gt_pos & pred_pos)
    fp = np.sum(~gt_pos & pred_pos)
    fn = np.sum(gt_pos & ~pred_pos)
    
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        "TP_sem": int(tp),
        "FP_sem": int(fp),
        "FN_sem": int(fn),
        "IOU_semantic": iou
    }

def compute_instance_metrics(gt_instances, pred_instances, matched_gt, matched_pred):
    """计算实例分割指标"""
    tp = len(matched_gt)
    fp = len(pred_instances) - len(matched_pred)
    fn = len(gt_instances) - len(matched_gt)
    
    return {
        "TP_inst": tp,
        "FP_inst": fp,
        "FN_inst": fn
    }

def compute_plot_metrics(gt_instances, pred_instances, matched_gt, matched_pred, matched_ious):
    """
    计算每个地块的Acc, Pre, Rec, F1, IoU
    根据论文公式：
    - Acc = (TP + TN) / (TP + TN + FP + FN)
    - Pre = TP / (TP + FP)
    - Rec = TP / (TP + FN)
    - IoU = 平均IoU（匹配的实例对）
    """
    tp = len(matched_gt)
    fp = len(pred_instances) - len(matched_pred)
    fn = len(gt_instances) - len(matched_gt)
    
    # 注意：在实例分割中，TN通常定义为0（因为每个实例要么是正例要么是负例）
    # 但对于每个地块，我们考虑所有实例，TN = 总实例数 - TP - FP - FN
    total_instances = len(gt_instances) + len(pred_instances) - len(matched_gt)
    # 实际上，对于实例分割，更常见的做法是只关注TP, FP, FN
    # 这里为了符合论文公式，我们计算Acc = TP / (TP + FP + FN)
    # 如果分母为0，则设为0
    
    # 计算平均IoU（仅计算匹配成功的实例对）
    avg_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    # 计算Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算Accuracy: TP / (TP + FP + FN) 
    # 这是实例分割中常用的accuracy定义
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": avg_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "matched_pairs": len(matched_ious)
    }

def print_metrics_summary(metrics_list, metric_name="地块"):
    """打印指标汇总统计"""
    if not metrics_list:
        print(f"没有{metric_name}的指标数据")
        return
    
    metrics_array = np.array(metrics_list)
    mean_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)
    
    print(f"\n{'='*60}")
    print(f"【{metric_name}指标汇总（平均值 ± 标准差）】")
    print(f"{'='*60}")
    print(f"Accuracy : {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall   : {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"F1-Score : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"IoU      : {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
    print(f"\n平均匹配对数: {np.mean([m[5] for m in metrics_list]):.2f}")
    print(f"平均TP数: {np.mean([m[6] for m in metrics_list]):.2f}")
    print(f"平均FP数: {np.mean([m[7] for m in metrics_list]):.2f}")
    print(f"平均FN数: {np.mean([m[8] for m in metrics_list]):.2f}")
    
    return mean_metrics, std_metrics

if __name__ == "__main__":
    txt_path = "./instance_seg/BTA_aggregate_result" 
    all_txt_files = get_all_files(txt_path)
    
    
    # 存储每个地块的指标
    plot_metrics_list = []
    semantic_iou_list = []
    
    # 用于记录每个地块的详细信息
    plot_details = []
    
    for idx, txt_file in enumerate(all_txt_files):
        print(f"\n{'#'*60}")
        print(f"处理第 {idx+1}/{len(all_txt_files)} 个地块: {os.path.basename(txt_file)}")
        print(f"{'#'*60}")
        
        df = load_point_cloud(txt_file)
        
        # 语义分割评估
        sem_metrics = compute_semantic_iou(df, target_sem=1)
        print("\n【语义分割评估】")
        for k, v in sem_metrics.items():
            print(f"{k}: {v}")
        semantic_iou_list.append(sem_metrics["IOU_semantic"])
        
        # 构建实例
        gt_instances = build_instance_points(df, "gt_sem", "gt_id", target_sem=1)
        pred_instances = build_instance_points(df, "pred_sem", "pred_id", target_sem=1)
        
        print(f"\nGT 实例数: {len(gt_instances)}")
        print(f"Pred 实例数: {len(pred_instances)}")
        
        # 匈牙利算法实例匹配
        matches, matched_gt, matched_pred, matched_ious = hungarian_match_instances(
            gt_instances, pred_instances, iou_threshold=0.5
        )
        
        print(f"\n匹配成功的实例对数: {len(matched_ious)}")
        if len(matched_ious) > 0:
            print(f"匹配实例的平均IoU: {np.mean(matched_ious):.4f}")
            print(f"匹配实例的IoU范围: [{np.min(matched_ious):.4f}, {np.max(matched_ious):.4f}]")
        
        print("\n【实例匹配结果（前5对）】")
        if len(matches) > 0:
            print(matches.head(min(5, len(matches))))
        
        # 计算地块级别的指标
        plot_metrics = compute_plot_metrics(
            gt_instances, pred_instances, matched_gt, matched_pred, matched_ious
        )
        
        print("\n【当前地块实例分割指标】")
        print(f"Accuracy : {plot_metrics['Accuracy']:.4f}")
        print(f"Precision: {plot_metrics['Precision']:.4f}")
        print(f"Recall   : {plot_metrics['Recall']:.4f}")
        print(f"F1-Score : {plot_metrics['F1']:.4f}")
        print(f"IoU      : {plot_metrics['IoU']:.4f}")
        print(f"TP: {plot_metrics['TP']}, FP: {plot_metrics['FP']}, FN: {plot_metrics['FN']}")
        print("=" * 60)
        
        # 存储指标
        plot_metrics_list.append([
            plot_metrics['Accuracy'],
            plot_metrics['Precision'],
            plot_metrics['Recall'],
            plot_metrics['F1'],
            plot_metrics['IoU'],
            plot_metrics['matched_pairs'],
            plot_metrics['TP'],
            plot_metrics['FP'],
            plot_metrics['FN']
        ])
        
        plot_details.append({
            'file': os.path.basename(txt_file),
            'gt_count': len(gt_instances),
            'pred_count': len(pred_instances),
            'matched': len(matched_ious),
            'tp': plot_metrics['TP'],
            'fp': plot_metrics['FP'],
            'fn': plot_metrics['FN'],
            'acc': plot_metrics['Accuracy'],
            'pre': plot_metrics['Precision'],
            'rec': plot_metrics['Recall'],
            'f1': plot_metrics['F1'],
            'iou': plot_metrics['IoU']
        })
    
    # 输出汇总结果
    print("\n" + "=" * 60)
    print("【最终汇总评估结果】")
    print("=" * 60)
    
    # 语义分割平均IoU
    print(f"\n语义分割平均IoU: {np.mean(semantic_iou_list):.4f} ± {np.std(semantic_iou_list):.4f}")
    
    # 实例分割指标汇总（地块间平均）
    mean_metrics, std_metrics = print_metrics_summary(plot_metrics_list, "地块")
    
    # 打印每个地块的详细结果
    print("\n" + "=" * 60)
    print("【各地块详细结果】")
    print("=" * 60)
    details_df = pd.DataFrame(plot_details)
    print(details_df.to_string(index=False))
    
    # 保存结果到CSV
    details_df.to_csv('plot_metrics_results_yolo_maped_points.csv', index=False)
    print("\n结果已保存到 plot_metrics_results_yolo_maped_points.csv")
