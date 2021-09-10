# TM-NET: Deep Generative Networks for Textured Meshes

This is a Python3 / Pytorch implementation of TM-NET.


- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training and Test](#training-and-test)

# Setup

To run this code you need the following:

- A machine with multiple GPUs(memory >= 12GB)

- Python packages in the requirements.txt
```
pip install -r requirements.txt
```

# Prepare Data

[Data Link](https://forms.gle/38Bn7MMpdhTmgffd6)

1. Run ```GetTransformedCube.m``` to get transformed a mini bounding box which will be used as source shape in non-rigid registration for each partial 3D model.
2. Run ```SupportAnalysis.m``` to extract structure information from the partial ```obj``` files producing a corresponding ```code.mat``` for each 3D model.
3. Run ```register.m``` to perform non-rigid registrations from transformed mini bounding boxes to original partial 3D models.
4. Run ```GenerateData.m``` to extract deformation information between source shapes and registered shapes which will be used as the input of ```TM-NET```.
5. Run ```TransferColorPerPixelScript.m``` to generate texture images for registered parts.
6. Run ```PrepareForTraining.m``` to split geometry, structure, image data to training or test dir. It will also divide texture image to six patches.

An example is shown in ```Pipeline.m```. After you successfully run the code, the dir structure will like follows:
```txt
├─box50
│  ├─37b6df64a97a5c29369151623ac3890b
│  └─d374912c3a9fca96c141a04b2a487fd9
├─Chair
│  ├─37b6df64a97a5c29369151623ac3890b
│  |   └─models
│  └─d374912c3a9fca96c141a04b2a487fd9
│      └─models
├─final50
│  ├─test
│  │  └─37b6df64a97a5c29369151623ac3890b
│  └─train
│      └─d374912c3a9fca96c141a04b2a487fd9
└─vaenew50
    ├─37b6df64a97a5c29369151623ac3890b
    │  ├─back
    │  ├─leg_ver_1
    │  ├─leg_ver_2
    │  ├─leg_ver_3
    │  ├─leg_ver_4
    │  └─seat
    └─d374912c3a9fca96c141a04b2a487fd9
        ├─back
        ├─leg_ver_1
        ├─leg_ver_2
        ├─leg_ver_3
        ├─leg_ver_4
        └─seat
```
Folder ```final50``` is all we need for training and test.

# Training and Test

- Train PartVAE for each part
```shell
python ./python/train.py --yaml ./python/yaml/table/surface/geovae.yml
python ./python/train.py --yaml ./python/yaml/table/leg/geovae.yml
```

- Train VQVAE
```shell
python ./python/train.py --yaml ./python/yaml/table/vqvae.yml
```

- Extract discrete code for the seed part
```shell
python ./python/extract_latents_central_part.py \
--image_dir ../data/table/ \
--mat_dir ../data/table \
--vqvae_ckpt ./table_vqvae/latest.pth \
--vqvae_yaml ./python/yaml/table/vqvae.yml \
--geovae_ckpt ./table_geovae/surface/latest.pth \
--geovae_yaml ./python/yaml/table/surface/geovae.yml \
--category table \
--save_path ../latent_data/table_latents \
--device 0 \
--mode 'train' or 'test' or 'val'
```

- Train conditional PixelSNAIL for the seed part
```shell
python ./python/train.py --yaml ./python/yaml/table/surface/pixelsnail_top.yml
python ./python/train.py --yaml ./python/yaml/table/surface/pixelsnail_bottom.yml
```

- Extract discrete code for other parts
```shell
python ./python/extract_latents_other_parts.py \
--image_dir ../data/table/ \
--mat_dir ../data/table \
--vqvae_ckpt ./table_vqvae/latest.pth \
--vqvae_yaml ./python/yaml/table/vqvae.yml \
--geovae_ckpt_dir ./table_geovae \
--geovae_yaml ./python/yaml/table/geovae.yml \
--category table \
--save_path ../latent_data/table_latents \
--device 0 \
--mode 'train' or 'test' or 'val'
```

- Train conditional PixelSNAIL for other parts
```shell
python ./python/train.py --yaml ./python/yaml/table/leg/pixelsnail_top.yml
python ./python/train.py --yaml ./python/yaml/table/leg/pixelsnail_bottom.yml
```

- Sample texture for the seed part
```shell
python ./python/conditional_sample_2levels_central_part.py \
--path ../latent_data/table_latents \
--part_name surface \
--vqvae ./table_vqvae/latest.pth \
--vqvae_yaml ./python/yaml/table/vqvae.yml \
--top ./table_pixelsnail/top_16/latest.pth \
--top_yaml ./python/yaml/table/pixelsnail_top_center_16.yml \
--bottom ./table_pixelsnail/bottom/latest.pth \
--bottom_yaml ./python/yaml/table/pixelsnail_bottom_center.yml \
--device 0 \
--batch 1
```

- Sample texture for other parts
```shell
python ./python/conditional_sample_2levels_other_parts.py \
--path ../latent_data/table_latents \
--central_part_name surface \
--part_name leg \
--vqvae ./table_vqvae/latest.pth \
--vqvae_yaml ./python/yaml/table/vqvae.yml \
--top ./table_pixelsnail/leg/top_16/latest.pth \
--top_yaml ./python/yaml/table/leg/pixelsnail_top_center_16.yml \
--bottom ./table_pixelsnail/leg/bottom/latest.pth \
--bottom_yaml ./python/yaml/table/leg/pixelsnail_bottom_center.yml \
--central_part_sample_dir ./table_pixelsnail/top_16/auto_texture \
--device 0 \
--batch 1
```