from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util_v2 import OliveBlockDataset, OliveWholeSceneDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
from collections import OrderedDict


classes_str = ['background', 'olive_tree']


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(args, io):

    # ============= Model ===================
    num_part = 2
    num_classes = 1
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part, points=args.num_points).to(device)
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    '''Resume or not'''
    if args.resume:
        state_dict = torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name,
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Resume training model...")
        print(torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name).keys())
    else:
        print("Training from scratch...")

    # =========== Dataloader =================
    train_data = OliveBlockDataset(root=args.data_root,
                                   npoints=args.num_points, split='train', color_channel=args.color,
                                   block_size=args.block_size, sample_rate=args.sample_rate,
                                   min_points=args.min_points)
    print("The number of training blocks is:%d", len(train_data))

    test_data = OliveBlockDataset(root=args.data_root,
                                  npoints=args.num_points, split='val', color_channel=args.color,
                                  block_size=args.block_size, sample_rate=args.val_sample_rate,
                                  min_points=args.min_points)
    print("The number of validation blocks is:%d", len(test_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0
    best_class_iou = 0
    best_instance_iou = 0

    for epoch in range(args.epochs):

        train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io)

        test_metrics, total_per_cat_iou = test_epoch(test_loader, model, epoch, num_part, num_classes, io)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        # 2. when get the best instance_iou, save the model:
        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            io.cprint('Max instance iou:%.5f' % best_instance_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou}
            torch.save(state, 'checkpoints/%s/best_insiou_model.pth' % args.exp_name)

        # 3. when get the best class_iou, save the model:
        # first we need to calculate the average per-class iou
        avg_class_iou = np.mean(total_per_cat_iou)
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            # print the iou of each class:
            for cat_idx in range(num_part):
                io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
            io.cprint('Max semantic iou:%.5f' % best_class_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou}
            torch.save(state, 'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max instance iou:%.5f' % best_instance_iou)
    io.cprint('Final Max class iou:%.5f' % best_class_iou)
    # save last model
    state = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou}
    torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))


def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (points, label, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points = Variable(points.float())
        label = Variable(label.long())
        target = Variable(target.long())
        norm_plt = points[:, :, 3:6]
        points = points[:, :, 0:3]
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

        # Note: We do not need to calculate per_class iou during training

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch+1, train_loss * 1.0 / count,
                                                                       metrics['accuracy'], metrics['shape_avg_iou'])
    io.cprint(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(num_part).astype(np.float32)
    final_total_per_cat_seen = np.zeros(num_part).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points = Variable(points.float())
        label = Variable(label.long())
        target = Variable(target.long())
        norm_plt = points[:, :, 3:6]
        points = points[:, :, 0:3]
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        pred_choice = seg_pred.max(dim=2)[1]
        for cls_idx in range(num_part):
            intersection = torch.logical_and(pred_choice == cls_idx, target == cls_idx).sum().item()
            union = torch.logical_or(pred_choice == cls_idx, target == cls_idx).sum().item()
            if union > 0:
                final_total_per_cat_iou[cls_idx] += intersection / float(union)
                final_total_per_cat_seen[cls_idx] += 1

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(num_part):
        if final_total_per_cat_seen[cat_idx] > 0:
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    io.cprint(outstr)

    return metrics, final_total_per_cat_iou


def load_eval_model(args, device, num_part=2):
    model = models.__dict__[args.model](num_part, points=args.num_points).to(device)
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if scale > 0:
        points = points / scale
    return points


def test(args, io):
    test_data = OliveWholeSceneDataset(root=args.data_root,
                                       block_points=args.num_points, split='test', color_channel=args.color,
                                       stride=args.stride, block_size=args.block_size, padding=args.padding)
    print("The number of test scenes is:%d", len(test_data))

    num_part = 2
    num_classes = 1
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_eval_model(args, device, num_part)
    io.cprint(str(model))

    metrics = defaultdict(lambda: list())
    total_per_cat_iou = np.zeros((num_part)).astype(np.float32)
    total_per_cat_seen = np.zeros((num_part)).astype(np.int32)
    all_scene_ious = []
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for scene_idx, (scene_blocks, scene_labels, scene_point_index) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        scene_blocks = scene_blocks.squeeze(0).numpy()
        scene_labels = scene_labels.squeeze(0).numpy()
        scene_point_index = scene_point_index.squeeze(0).numpy()
        whole_scene_target = test_data.semantic_labels_list[scene_idx]
        vote_label_pool = np.zeros((whole_scene_target.shape[0], num_part), dtype=np.int64)
        num_blocks = scene_blocks.shape[0]
        num_batches = int(np.ceil(num_blocks / float(args.test_batch_size)))
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.test_batch_size
                end_idx = min((batch_idx + 1) * args.test_batch_size, num_blocks)
                real_batch_size = end_idx - start_idx
                points_np = scene_blocks[start_idx:end_idx]
                index_np = scene_point_index[start_idx:end_idx]
                label_np = np.zeros((real_batch_size,), dtype=np.int64)
                points = torch.Tensor(points_np[:, :, 0:3]).float().to(device)
                norm_plt = torch.Tensor(points_np[:, :, 3:6]).float().to(device)
                label = torch.Tensor(label_np).long().to(device)
                seg_pred = model(points.transpose(2, 1), norm_plt.transpose(2, 1), to_categorical(label, num_classes))
                pred_label = seg_pred.cpu().data.max(2)[1].numpy()
                for b in range(real_batch_size):
                    for n in range(pred_label.shape[1]):
                        vote_label_pool[int(index_np[b, n]), int(pred_label[b, n])] += 1

        pred_label = np.argmax(vote_label_pool, axis=1).astype(np.int32)
        per_class_iou = []
        for cls_idx in range(num_part):
            intersection = np.sum((pred_label == cls_idx) & (whole_scene_target == cls_idx))
            union = np.sum((pred_label == cls_idx) | (whole_scene_target == cls_idx))
            if union > 0:
                iou = intersection / float(union)
                total_per_cat_iou[cls_idx] += iou
                total_per_cat_seen[cls_idx] += 1
                per_class_iou.append(iou)
        all_scene_ious.append(np.mean(per_class_iou) if len(per_class_iou) else 0.0)
        scene_acc = np.mean(pred_label == whole_scene_target)
        metrics['accuracy'].append(scene_acc)

    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['shape_avg_iou'] = np.mean(all_scene_ious)
    for cat_idx in range(num_part):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    class_iou = 0
    for cat_idx in range(num_part):
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
    avg_class_iou = class_iou / num_part
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


def inference(args, io):
    num_part = 2
    num_classes = 1
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_eval_model(args, device, num_part)
    if args.inference_input_dir is None:
        inference_root = args.data_root
        inference_split = args.inference_split
    else:
        input_dir = os.path.abspath(os.path.normpath(args.inference_input_dir))
        inference_root = os.path.dirname(input_dir)
        inference_split = os.path.basename(input_dir)
    output_dir = args.inference_output_dir or os.path.join('checkpoints', args.exp_name, 'inference_' + inference_split)
    os.makedirs(output_dir, exist_ok=True)
    test_data = OliveWholeSceneDataset(root=inference_root,
                                       block_points=args.num_points, split=inference_split, color_channel=args.color,
                                       stride=args.stride, block_size=args.block_size, padding=args.padding)
    io.cprint('Inference scenes: %d' % len(test_data))
    for scene_idx in tqdm(range(len(test_data)), total=len(test_data), smoothing=0.9):
        whole_scene_data = test_data.scene_data_list[scene_idx]
        vote_label_pool = np.zeros((test_data.semantic_labels_list[scene_idx].shape[0], num_part), dtype=np.int64)
        scene_data, _, scene_point_index = test_data[scene_idx]
        num_blocks = scene_data.shape[0]
        num_batches = int(np.ceil(num_blocks / float(args.test_batch_size)))
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * args.test_batch_size
                end_idx = min((batch_idx + 1) * args.test_batch_size, num_blocks)
                real_batch_size = end_idx - start_idx
                points_np = scene_data[start_idx:end_idx]
                index_np = scene_point_index[start_idx:end_idx]
                label_np = np.zeros((real_batch_size,), dtype=np.int64)
                points = torch.Tensor(points_np[:, :, 0:3]).float().to(device)
                norm_plt = torch.Tensor(points_np[:, :, 3:6]).float().to(device)
                label = torch.Tensor(label_np).long().to(device)
                seg_pred = model(points.transpose(2, 1), norm_plt.transpose(2, 1), to_categorical(label, num_classes))
                pred_label = seg_pred.cpu().data.max(2)[1].numpy()
                for b in range(real_batch_size):
                    for n in range(pred_label.shape[1]):
                        vote_label_pool[int(index_np[b, n]), int(pred_label[b, n])] += 1
        pred_label = np.argmax(vote_label_pool, axis=1).astype(np.int32)
        pred_data = np.concatenate([whole_scene_data, pred_label.reshape(-1, 1)], axis=1)
        scene_name = os.path.splitext(test_data.file_list[scene_idx])[0]
        np.savetxt(os.path.join(output_dir, '%s_with_pred.txt' % scene_name), pred_data,
                   fmt='%.6f %.6f %.6f %.0f %.0f %.0f %.0f %.0f %.0f')
    io.cprint('Inference results saved to %s' % output_dir)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP_olive')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--inference', type=bool, default=False,
                        help='run inference and save point-wise predictions')
    parser.add_argument('--data_root', type=str, default='/workspace/olive_data_3_new',
                        help='dataset root containing train, val and test directories')
    parser.add_argument('--inference_split', type=str, default='test',
                        help='olive data split to run inference when inference_input_dir is not set')
    parser.add_argument('--inference_input_dir', type=str, default=None,
                        help='directory containing txt files for inference')
    parser.add_argument('--inference_output_dir', type=str, default=None,
                        help='directory to save txt files with appended pred_semantic')
    parser.add_argument('--inference_rounds', type=int, default=8,
                        help='number of random voting rounds for each inference file')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points per block to use')
    parser.add_argument('--block_size', type=float, default=4.0,
                        help='sliding block size for train/val blocks')
    parser.add_argument('--stride', type=float, default=2.0,
                        help='sliding stride for whole-scene inference and test')
    parser.add_argument('--padding', type=float, default=0.001,
                        help='padding around block window during whole-scene inference and test')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='sampling rate for train blocks')
    parser.add_argument('--val_sample_rate', type=float, default=1.0,
                        help='sampling rate for val blocks')
    parser.add_argument('--min_points', type=int, default=128,
                        help='minimum points per sampled block')
    parser.add_argument('--color', type=bool, default=True,
                        help='use rgb channel as additional features')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name

    _init_()

    if args.inference:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_inference.log' % (args.exp_name))
    elif not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if args.inference:
        inference(args, io)
    elif not args.eval:
        train(args, io)
    else:
        test(args, io)
