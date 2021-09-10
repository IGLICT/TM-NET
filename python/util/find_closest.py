# python find_closest.py \
# --shapenet_dir /mnt/f/wutong/data/car_new_reg \
# --train_lst /mnt/f/wutong/data/car_new_reg/train.lst \
# --tgt_dir '' \
# --mode ACAP \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/b/wutong/222/Data/ShapeNetCore.v1/02958343 \
# --train_lst /mnt/f/wutong/data/car_new_reg/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/car \
# --mode CD \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/sdg/wutong/65e/wutong/ShapeNetCore.v2/03001627 \
# --train_lst /mnt/f/wutong/data/chair/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/chair \
# --mode CD \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/sdg/wutong/65e/lmx/siga20/data/models/plane/transferred_dir \
# --train_lst /mnt/f/wutong/data/plane/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/plane \
# --mode CD \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/sdg/wutong/65e/lmx/siga20/data/models/04379243_table/transferred_dir \
# --train_lst /mnt/f/wutong/data/table/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/table \
# --mode CD \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Data/20210508_RenderedImagesBlender/04379243_table \
# --train_lst /mnt/f/wutong/data/table/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Data/20210508_RenderedImagesBlenderGenerated/04379243_table \
# --mode img \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Data/20210508_RenderedImagesBlender/03001627_chair \
# --train_lst /mnt/f/wutong/data/chair/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/3_Interpolation_Compare/VON_interpolation/selected \
# --mode single_img \
# --device -1

# python find_closest.py \
# --shapenet_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Data/20210508_RenderedImagesBlender/02958343_car \
# --train_lst /mnt/f/wutong/data/car/train.lst \
# --tgt_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/3_Interpolation_Compare/VON_interpolation/car/selected \
# --mode single_img \
# --device -1

import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d
import trimesh
import torch
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

from chamferdist import ChamferDistance
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

def load_acap(path):
    geo_data = sio.loadmat(path, verify_compressed_data_integrity=False)
    try:
        LOGR = geo_data['fmlogdr']
        S = geo_data['fms']
    except:
        print(path)
    
    if LOGR.shape[0] == 1:
        LOGR = np.squeeze(LOGR, axis=0)
    if S.shape[0] == 1:
        S = np.squeeze(S, axis=0)
    origin_geo_input = np.concatenate((LOGR, S), axis = 1)
    return origin_geo_input

def readOBJ(filepath):
    """
    READOBJ read .obj file
    Input:
      filepath a string of mesh file path
    Output:
      V (|V|,3) numpy array of vertex positions
	  F (|F|,3) numpy array of face indices
    """
    V = []
    F = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    while True:
        for line in lines:
            if line == "":
                break
            elif line.strip().startswith("vn"):
                continue
            elif line.strip().startswith("vt"):
                continue
            elif line.strip().startswith("v"):
                vertices = line.replace("\n", "").split(" ")[1:]
                vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                V.append(list(map(float, vertices)))
            elif line.strip().startswith("f"):
                t_index_list = []
                for t in line.replace("\n", "").split(" ")[1:]:
                    t_index = t.split("/")[0]
                    try: 
                        t_index_list.append(int(t_index) - 1)
                    except ValueError:
                        continue
                F.append(t_index_list)
            else:
                continue
        break
    V = np.asarray(V)
    F = np.asarray(F)
    return V, F

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--shapenet_dir', type=str, required=True, default='')
    parser.add_argument('--train_lst', type=str, default=None)
    parser.add_argument('--tgt_dir', type=str, default='./query_img')
    parser.add_argument('--mode', type=str, default='CD')
    parser.add_argument('--device', type=int, default=-1)
    opt = parser.parse_args()
    opt = parser.parse_args()
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    opt.device = num2device_dict[opt.device]
    print(opt.device)

    chamferDist = ChamferDistance()
    if opt.mode == 'CD':
        N = 1024

        shapenet_dir = opt.shapenet_dir
        train_lst = opt.train_lst
        tgt_dir = opt.tgt_dir

        
        tgt_obj_files = glob.glob(os.path.join(tgt_dir, 'F13*', 'merge1.obj'))
        # tgt_obj_files = glob.glob(os.path.join(tgt_dir, 'F13*', '*', 'merge2.obj'))
        N_tgt = len(tgt_obj_files)
        print(N_tgt)
        
        if N_tgt == 0:
            return
        tgt_clouds = torch.zeros(N_tgt, N, 3)
        dist_dicts = [{} for i in range(N_tgt)]
        dist_jsons = []
        for i, tgt_obj_file in enumerate(tgt_obj_files):
            # tgt_obj_file = '/mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/car/dec/3_done/merge1.obj'
            dist_jsons.append(os.path.join(os.path.dirname(tgt_obj_file), 'chamfer.json'))
            # tgt_obj = o3d.io.read_triangle_mesh(tgt_obj_file)
            # tgt_pc = tgt_obj.sample_points_uniformly(number_of_points=N)
            # tgt_cloud = torch.from_numpy(np.asarray(tgt_pc.points)).to(torch.float).to(torch.device(opt.device))

            tgt_obj = trimesh.load_mesh(tgt_obj_file)
            tgt_pc, face_index = trimesh.sample.sample_surface(tgt_obj, N)
            tgt_cloud = torch.from_numpy(np.asarray(tgt_pc)).to(torch.float).to(torch.device(opt.device))
            
            tgt_clouds[i, :, :] = tgt_cloud

        train_ids = np.loadtxt(train_lst, dtype=str)
        train_ids = sorted(train_ids)
        
        for i, train_id in enumerate(train_ids):
            # if i % 3 != 0:
            #     continue
            # car
            # src_obj_file = os.path.join(shapenet_dir, train_id, 'model.obj')
            # chair
            # src_obj_file = os.path.join(shapenet_dir, train_id, 'models', 'model_normalized.obj')
            # plane
            # src_obj_file = os.path.join(shapenet_dir, train_id, 'models', 'model_normalized.obj')
            # table
            src_obj_file = os.path.join(shapenet_dir, train_id, 'model.obj')
            
            if os.path.exists(src_obj_file):
                print(src_obj_file)
                try:
                    # src_obj = trimesh.load_mesh(src_obj_file)
                    # src_pc, face_index = trimesh.sample.sample_surface(src_obj, N)
                    # src_cloud = torch.from_numpy(np.asarray(src_pc)).to(torch.float).to(torch.device(opt.device))

                    # src_obj = o3d.io.read_triangle_mesh(src_obj_file)
                    # src_pc = src_obj.sample_points_uniformly(number_of_points=N)
                    # src_cloud = torch.from_numpy(np.asarray(src_pc.points)).to(torch.float).to(torch.device(opt.device))
                    V, F  = readOBJ(src_obj_file)
                    src_obj = o3d.geometry.TriangleMesh()
                    src_obj.vertices = o3d.utility.Vector3dVector(V)
                    src_obj.triangles = o3d.utility.Vector3iVector(F)
                    src_pc = src_obj.sample_points_uniformly(number_of_points=N)
                    src_cloud = torch.from_numpy(np.asarray(src_pc.points)).to(torch.float).to(torch.device(opt.device))
                
                    # dist_forward = chamferDist(src_cloud.unsqueeze(0), tgt_cloud.unsqueeze(0))
                    # src_clouds = src_cloud.repeat(N_tgt, 1, 1)
                    # dist_forward = chamferDist(src_clouds, tgt_clouds)
                    src_cloud = src_cloud.unsqueeze(0)
                    for j in range(N_tgt):
                        dist_forward = chamferDist(src_cloud, tgt_clouds[j:j+1, :, :])
                        dist_dicts[j][src_obj_file] = dist_forward.detach().cpu().item()
                        # print(dist_forward.detach().cpu().item())
                except Exception as e:
                    print(e)
        for i in range(N_tgt):
            # sort
            sorted_dist_dict = {k: v for k, v in sorted(dist_dicts[i].items(), key=lambda item: item[1])}
            # print(sorted_dist_dict)
            with open(dist_jsons[i], 'w') as outfile:
                json.dump(sorted_dist_dict, outfile)

    elif opt.mode == 'img':
        shapenet_dir = opt.shapenet_dir
        train_lst = opt.train_lst
        tgt_dir = opt.tgt_dir

        sub_tgt_dirs = glob.glob(os.path.join(tgt_dir, '*'))
        H = 256
        W = 256
        V = 6
        C = 3
        N = len(sub_tgt_dirs)
        print(N)
        if 0 == N:
            return 0
        tgt_imgs = np.zeros((N, V, H, W, C))
        dist_dicts = [{} for i in range(N)]
        dist_jsons = []
        for i, sub_tgt_dir in enumerate(sub_tgt_dirs):
            dist_jsons.append(os.path.join(sub_tgt_dir, 'multiview.json'))
            for j in range(V):
                path = os.path.join(sub_tgt_dir, 'image', '00{}.png'.format(j))
                tgt_img = cv2.imread(path)
                tgt_imgs[i, j, :, :, :] = tgt_img

        train_ids = np.loadtxt(train_lst, dtype=str)
        train_ids = sorted(train_ids)
        for i, train_id in enumerate(train_ids):
            src_imgs = np.zeros((V, H, W, C))
            if not os.path.exists(os.path.join(shapenet_dir, train_id, 'image', '005.png')):
                continue
            print(train_id)
            for j in range(V):
                path = os.path.join(shapenet_dir, train_id, 'image', '00{}.png'.format(j))
                src_img = cv2.imread(path)
                src_imgs[j, :, :, :] = src_img
            # l2
            # diff = np.abs(tgt_imgs - src_imgs)
            # dist = np.sum(diff.reshape(diff.shape[0], -1), axis=-1)
            # for j in range(N):
            #     dist_dicts[j][path] = dist[j]

            # SSIM
            for j in range(N):
                temp = tgt_imgs[j, :, :, :, :]
                diff = ssim(np.reshape(temp, (V*256, 256, 3)), np.reshape(src_imgs, (V*256, 256, 3)), multichannel=True)
                dist_dicts[j][path] = -diff

        for i in range(N):
            # sort
            sorted_dist_dict = {k: v for k, v in sorted(dist_dicts[i].items(), key=lambda item: item[1])}
            # print(sorted_dist_dict)
            with open(dist_jsons[i], 'w') as outfile:
                json.dump(sorted_dist_dict, outfile)
    elif opt.mode == 'single_img':
        shapenet_dir = opt.shapenet_dir
        train_lst = opt.train_lst
        tgt_dir = opt.tgt_dir

        sub_tgt_dirs = glob.glob(os.path.join(tgt_dir, '*'))
        H = 128
        W = 128
        V = 1
        C = 3
        N = len(sub_tgt_dirs)
        print(N)
        if 0 == N:
            return 0
        tgt_imgs = np.zeros((N, V, H, W, C))
        dist_dicts = [{} for i in range(N)]
        dist_jsons = []
        for i, sub_tgt_dir in enumerate(sub_tgt_dirs):
            dist_jsons.append(os.path.join(sub_tgt_dir, 'multiview.json'))
            for j in range(V):
                path = os.path.join(sub_tgt_dir, '002.jpg')
                if not os.path.exists(path):
                    print('{} not exists'.format(path))
                tgt_img = cv2.imread(path)
                tgt_imgs[i, j, :, :, :] = tgt_img

        train_ids = np.loadtxt(train_lst, dtype=str)
        train_ids = sorted(train_ids)
        for i, train_id in enumerate(train_ids):
            if i % 10 != 0:
                continue
            src_imgs = np.zeros((V, H, W, C))
            print(train_id)
            exist_flag = True
            for j in range(V):
                path = os.path.join(shapenet_dir, train_id, 'image', '003.png')
                if not os.path.exists(path):
                    print('{} not exists'.format(path))
                    exist_flag = False
                else:
                    src_img = cv2.imread(path)
                    src_imgs[j, :, :, :] = cv2.resize(src_img, (H, W))
            if exist_flag == False:
                continue
            # SSIM
            for j in range(N):
                temp = tgt_imgs[j, :, :, :, :]
                diff = ssim(np.reshape(temp, (V*H, W, 3)), np.reshape(src_imgs, (V*H, W, 3)), multichannel=True)
                
                dist_dicts[j][path] = -diff
            # print(diff)

        for i in range(N):
            # sort
            sorted_dist_dict = {k: v for k, v in sorted(dist_dicts[i].items(), key=lambda item: item[1])}
            # print(sorted_dist_dict)
            with open(dist_jsons[i], 'w') as outfile:
                json.dump(sorted_dist_dict, outfile)
    elif opt.mode == 'ACAP':
        query_ids = ['12a11a800f6869d99ea5c205086e2a63',
                    '15bdc3b07627ea2f6dca8796028a1679',
                    '1a64bf1e658652ddb11647ffa4306609',
                    '1dada716fbc92664a1b543d55837199',
                    '26b49835f02327b283ffcfc40c29975',
                    '2a07432c2860b54becf71e2e014ff6f',
                    '38b76f9ebe07ff52ce638347977b4bb9',
                    '3b2d9c065622422c373d55cb23dc8dea',
                    '3ef0d561be7aeb3080275d4a88484513',
                    '5a728c7371f8fa3f1e0401312b984eca',
                    '6acb271397e6f9b9e858db1dc3499392',
                    '7e3237619d96b6e551a95aaa6caba1d3',
                    '8a581532d98283f547c236339706f200',
                    '8cbc8e27e885eb763ffe0e5069bf1eb5',
                    '9f3c3a9a183818a1f200a72c9245aee7',]
        part_name = 'body'
        
        train_lst = opt.train_lst
        train_ids = np.loadtxt(train_lst, dtype=str)
        N = len(query_ids)

        dist_dicts = [{} for i in range(N)]
        dist_jsons = []
        query_mats = []
        for i, query_id in enumerate(query_ids):
            dist_json = os.path.join('.', '{}.json'.format(query_id))
            dist_jsons.append(dist_json)

            path = os.path.join(opt.shapenet_dir, query_id, '{}_{}.mat'.format(query_id, part_name))
            origin_geo_input = load_acap(path)
            query_mats.append(origin_geo_input)

        query_mats = np.array(query_mats)
        for i, train_id in enumerate(train_ids):
            print(train_id)
            path = os.path.join(opt.shapenet_dir, train_id, '{}_{}.mat'.format(train_id, part_name))
            if not os.path.exists(path):
                continue
            origin_geo_input = load_acap(path)

            diff = np.abs(query_mats - origin_geo_input)
            dist = np.sum(diff.reshape(diff.shape[0], -1), axis=-1)
            for j in range(N):
                dist_dicts[j][path] = dist[j]
        for i in range(N):
            # sort
            sorted_dist_dict = {k: v for k, v in sorted(dist_dicts[i].items(), key=lambda item: item[1])}
            # print(sorted_dist_dict)
            with open(dist_jsons[i], 'w') as outfile:
                json.dump(sorted_dist_dict, outfile)

if __name__ == '__main__':
    main()