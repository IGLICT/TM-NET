import glob
import os
import numpy as np
import cv2

def merge_patches(patch_image_dir, channel):
    patch0_files = glob.glob(os.path.join(patch_image_dir, '*patch0.png*'))
    merged_dir = os.path.join(patch_image_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)

    for patch0_file in patch0_files:
        head_tail = os.path.split(patch0_file)
        head = head_tail[0]
        tail = head_tail[1]

        # if already exists, just jump over it.
        out_name = os.path.join(merged_dir, tail.replace('patch0', ''))
        if os.path.exists(out_name):
            continue

        patch0 = cv2.imread(patch0_file, cv2.IMREAD_UNCHANGED)
        # get filename
        patch1_file = os.path.join(head, tail.replace('patch0', 'patch1'))
        patch2_file = os.path.join(head, tail.replace('patch0', 'patch2'))
        patch3_file = os.path.join(head, tail.replace('patch0', 'patch3'))
        patch4_file = os.path.join(head, tail.replace('patch0', 'patch4'))
        patch5_file = os.path.join(head, tail.replace('patch0', 'patch5'))
        # read
        patch1 = cv2.imread(patch1_file, cv2.IMREAD_UNCHANGED)
        patch2 = cv2.imread(patch2_file, cv2.IMREAD_UNCHANGED)
        patch3 = cv2.imread(patch3_file, cv2.IMREAD_UNCHANGED)
        patch4 = cv2.imread(patch4_file, cv2.IMREAD_UNCHANGED)
        patch5 = cv2.imread(patch5_file, cv2.IMREAD_UNCHANGED)
        # indices
        H_begin = [256, 0, 256, 512, 768, 256]
        W_begin = [0, 256, 256, 256, 256, 512]
        patches = []
        patches.append(patch0)
        patches.append(patch1)
        patches.append(patch2)
        patches.append(patch3)
        patches.append(patch4)
        patches.append(patch5)
        # merge
        merged_image = np.zeros((768, 1024, channel), np.uint8)
        for i in range(6):
            try:
                merged_image[W_begin[i]:W_begin[i]+256, H_begin[i]:H_begin[i]+256, :] = patches[i]
            except Exception:
                continue
        # save
        cv2.imwrite(out_name, merged_image)

        # rm
        # cmd = 'rm {}/*png'.format(patch_image_dir)
        # os.system(cmd)


