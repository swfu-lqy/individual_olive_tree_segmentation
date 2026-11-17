import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def load_olive_scene_file(path):
    if path.lower().endswith('.ply'):
        data = load_olive_ply(path)
    else:
        data = np.loadtxt(path).astype(np.float32)
        if data.shape[1] == 6:
            data = np.concatenate([data, np.zeros((data.shape[0], 2), dtype=np.float32)], axis=1)
        elif data.shape[1] == 7:
            data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            data = data[:, :8]
    return data.astype(np.float32)


def load_olive_ply(path):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Invalid ply file: %s' % path)
            text = line.decode('utf-8', errors='ignore').strip()
            header.append(text)
            if text == 'end_header':
                break
        header_bytes = f.tell()
    fmt = None
    vertex_count = None
    properties = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'format':
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == 'element':
            in_vertex = parts[1] == 'vertex'
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == 'property':
            properties.append((parts[1], parts[-1]))
    if vertex_count is None:
        raise ValueError('No vertex element found in ply file: %s' % path)
    prop_names = [name for _, name in properties]
    required = ['x', 'y', 'z']
    if not all(name in prop_names for name in required):
        raise ValueError('PLY must contain x y z properties: %s' % path)
    color_names = [('red', 'green', 'blue'), ('r', 'g', 'b')]
    color_props = next((names for names in color_names if all(name in prop_names for name in names)), None)
    if color_props is None:
        raise ValueError('PLY must contain red/green/blue or r/g/b properties: %s' % path)
    if fmt == 'ascii':
        vertices = np.loadtxt(path, skiprows=len(header), max_rows=vertex_count).astype(np.float32)
        xyz = vertices[:, [prop_names.index('x'), prop_names.index('y'), prop_names.index('z')]]
        rgb = vertices[:, [prop_names.index(color_props[0]), prop_names.index(color_props[1]), prop_names.index(color_props[2])]]
    else:
        endian = '<' if fmt == 'binary_little_endian' else '>'
        type_map = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1', 'short': 'i2', 'ushort': 'u2',
                    'int16': 'i2', 'uint16': 'u2', 'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
                    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}
        dtype = np.dtype([(name, endian + type_map[prop_type]) for prop_type, name in properties])
        with open(path, 'rb') as f:
            f.seek(header_bytes)
            vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
        rgb = np.vstack([vertices[color_props[0]], vertices[color_props[1]], vertices[color_props[2]]]).T.astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    semantic = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    instance = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, rgb, semantic, instance], axis=1)


# =========== ModelNet40 =================
class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition  # Here the new given partition will cover the 'train'

    def __getitem__(self, item):  # indice of the pts or label
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = pc_normalize(pointcloud)  # you can try to add it or not to train our model
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle the order of pts
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False):
        self.npoints = npoints
        self.root = '/workspace/pointMLP-pytorch-main/part_segmentation/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)


class OliveBlockDataset(Dataset):
    def __init__(self, root='./../data/olive_data', npoints=4096,
                 split='train', color_channel=True, block_size=4.0, sample_rate=1.0, min_points=128, cache_size=20000):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.color_channel = color_channel
        self.block_size = block_size
        self.min_points = min_points
        self.cache_size = cache_size
        self.cache = {}
        split_dir = os.path.join(self.root, split)
        self.datapath = sorted([os.path.join(split_dir, fn) for fn in os.listdir(split_dir) if fn.endswith('.txt')])
        self.scene_points_list = []
        self.semantic_labels_list = []
        num_point_all = []
        for path in self.datapath:
            data = np.loadtxt(path).astype(np.float32)
            self.scene_points_list.append(data[:, 0:6])
            self.semantic_labels_list.append(data[:, 6].astype(np.int32))
            num_point_all.append(data.shape[0])
        sample_prob = np.array(num_point_all) / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / npoints)
        scene_idxs = []
        for index in range(len(self.datapath)):
            scene_idxs.extend([index] * max(1, int(round(sample_prob[index] * num_iter))))
        self.scene_idxs = np.array(scene_idxs)
        self.classes = {'Plot': 0}
        self.seg_classes = {'Plot': [0, 1]}

    def __getitem__(self, index):
        scene_idx = self.scene_idxs[index]
        points = self.scene_points_list[scene_idx]
        labels = self.semantic_labels_list[scene_idx]
        if index in self.cache:
            point_idxs = self.cache[index]
        else:
            point_idxs = None
            for _ in range(10):
                center = points[np.random.choice(points.shape[0]), 0:3]
                block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
                candidate_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                                          (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                if candidate_idxs.size >= self.min_points:
                    point_idxs = candidate_idxs
                    break
            if point_idxs is None:
                point_idxs = np.arange(points.shape[0])
            if len(self.cache) < self.cache_size:
                self.cache[index] = point_idxs
        replace = point_idxs.size < self.npoints
        selected_idxs = np.random.choice(point_idxs, self.npoints, replace=replace)
        point_set = points[selected_idxs, 0:6].copy() if self.color_channel else points[selected_idxs, 0:3].copy()
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.color_channel:
            point_set[:, 3:6] = point_set[:, 3:6] / 255.0
        cls = np.array([0]).astype(np.int32)
        seg = labels[selected_idxs]
        return point_set, cls, seg

    def __len__(self):
        return len(self.scene_idxs)


class OliveWholeSceneDataset:
    def __init__(self, root='./../data/olive_data', block_points=4096,
                 split='test', color_channel=True, stride=2.0, block_size=4.0, padding=0.001):
        self.root = root
        self.block_points = block_points
        self.split = split
        self.color_channel = color_channel
        self.stride = stride
        self.block_size = block_size
        self.padding = padding
        split_dir = os.path.join(self.root, split)
        self.file_list = sorted([fn for fn in os.listdir(split_dir) if fn.endswith('.txt') or fn.endswith('.ply')])
        self.datapath = [os.path.join(split_dir, fn) for fn in self.file_list]
        self.scene_data_list = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.instance_labels_list = []
        for path in self.datapath:
            data = load_olive_scene_file(path)
            self.scene_data_list.append(data)
            self.scene_points_list.append(data[:, 0:6])
            self.semantic_labels_list.append((data[:, 6] > 0).astype(np.int32))
            self.instance_labels_list.append(data[:, 7].astype(np.int32) if data.shape[1] > 7 else np.zeros(data.shape[0], dtype=np.int32))

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        coord_min = np.amin(points[:, 0:3], axis=0)
        coord_max = np.amax(points[:, 0:3], axis=0)
        grid_x = max(1, int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1))
        grid_y = max(1, int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1))
        data_room, label_room, index_room = [], [], []
        for index_y in range(grid_y):
            for index_x in range(grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) &
                                      (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_blocks = int(np.ceil(point_idxs.size / float(self.block_points)))
                point_size = num_blocks * self.block_points
                if point_size > point_idxs.size:
                    point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=True)
                    point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                for block_idx in range(num_blocks):
                    block_point_idxs = point_idxs[block_idx * self.block_points:(block_idx + 1) * self.block_points]
                    block_points = points[block_point_idxs, 0:6].copy() if self.color_channel else points[block_point_idxs, 0:3].copy()
                    block_points[:, 0:3] = pc_normalize(block_points[:, 0:3])
                    if self.color_channel:
                        block_points[:, 3:6] = block_points[:, 3:6] / 255.0
                    data_room.append(block_points)
                    label_room.append(labels[block_point_idxs])
                    index_room.append(block_point_idxs)
        if len(data_room) == 0:
            point_idxs = np.arange(points.shape[0])
            point_size = int(np.ceil(point_idxs.size / float(self.block_points))) * self.block_points
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=True)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            for block_idx in range(point_size // self.block_points):
                block_point_idxs = point_idxs[block_idx * self.block_points:(block_idx + 1) * self.block_points]
                block_points = points[block_point_idxs, 0:6].copy() if self.color_channel else points[block_point_idxs, 0:3].copy()
                block_points[:, 0:3] = pc_normalize(block_points[:, 0:3])
                if self.color_channel:
                    block_points[:, 3:6] = block_points[:, 3:6] / 255.0
                data_room.append(block_points)
                label_room.append(labels[block_point_idxs])
                index_room.append(block_point_idxs)
        return np.stack(data_room, axis=0), np.stack(label_room, axis=0), np.stack(index_room, axis=0)

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # train_data = PartNormalDataset(npoints=2048, split='train', normalize=False)
    train_data = OliveSemanticDataset(npoints=65536, split='test', normalize=False)
    print(len(train_data))
    # test = PartNormalDataset(npoints=2048, split='test', normalize=False)
    for data, label, seg, normal in train_data:
        print(data.shape)
        output_data = np.hstack([data, normal, seg.reshape(-1, 1)])  # 将标签和颜色添加到原始数据
        np.savetxt("point_cloud_random.txt", output_data, fmt='%f %f  %f  %d  %d  %d  %d')
        print(np.unique(label))
        print(np.unique(seg))
        print(normal.shape)
        break
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0,
                              drop_last=True)
    for batch_id, (points, label, target, norm_plt) in enumerate(train_loader):
        print(batch_id, points.shape, label.shape, target.shape, norm_plt.shape)
        break
