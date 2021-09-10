# python copy_dir.py \
# --json_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/car/F13_3_done \
# --shapenet_dir /mnt/sdg/wutong/65e/lmx/siga20/data/models/car/transferred_dir

# python copy_dir.py \
# --json_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/chair/F13_3_done \
# --shapenet_dir /mnt/sdg/wutong/65e/wutong/ShapeNetCore.v2/03001627

# python copy_dir.py \
# --json_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/plane/F13_3_done/f16381a160f20bc4a3b534252984039 \
# --shapenet_dir /mnt/sdg/wutong/65e/lmx/siga20/data/models/plane/transferred_dir

# python copy_dir.py \
# --json_dir /mnt/b/wutong/222/PapersAndCode/Siggraph2020/TEXTURE/Example_SIG21_revision/2_Random/selected/table/F13_3_done \
# --shapenet_dir /mnt/sdg/wutong/65e/wutong/ShapeNetCore.v2/04379243

import json
import os
import glob
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_dir', type=str, required=True, default='')
    parser.add_argument('--shapenet_dir', type=str, required=True, default='')
    opt = parser.parse_args()

    json_dir = opt.json_dir
    json_files = glob.glob(os.path.join(json_dir, '*multiview.json'))

    for json_file in json_files:
        with open(json_file) as f:
            dist_dict = json.load(f)
        model_id = os.path.basename(json_file).replace('.json', '')
        sub_dir = os.path.join(json_dir, model_id)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        for i, (k, v) in enumerate(dist_dict.items()):
            if i <= 10:
                # cur_id = os.path.basename(k).split('_')[0]
                cur_id = os.path.basename(os.path.dirname(os.path.dirname(k)))
                # cur_id = os.path.basename(os.path.dirname(k))
                cmd = 'cp -r {} {}'.format(os.path.join(opt.shapenet_dir, cur_id), os.path.join(sub_dir, '{}_{}').format(i, cur_id))
                os.system(cmd)
