# train vqvae for all parts
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/vqvae.yml

# train geovae for each part
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/surface/geovae.yml
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/leg/geovae.yml

# extract latents from geovae and vqvae for the seed part
python code_TMNET/extract_latents_central_part_aug.py \
--image_dir /mnt/f/wutong/data/table/ \
--mat_dir /mnt/f/wutong/data/table \
--vqvae_ckpt ./20210601_table/latest1.pth \
--vqvae_yaml ./code_TMNET/yaml/table/vqvae.yml \
--geovae_ckpt ./20210601_table_geo/leg/latest1.pth \
--geovae_yaml ./code_TMNET/yaml/table/surface/geovae.yml \
--category table \
--save_path ./20210601_table_latents \
--device 0 \
--mode 'train'/'test'/'val'

# train top and bottom level pixelsnail for the seed part
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/surface/pixelsnail_top.yml
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/surface/pixelsnail_bottom.yml

# extract latents from geovae and vqvae for other parts
python code_TMNET/extract_latents_other_parts.py \
--image_dir /mnt/f/wutong/data/table/ \
--mat_dir /mnt/f/wutong/data/table \
--vqvae_ckpt ./20210601_table/latest1.pth \
--vqvae_yaml ./code_TMNET/yaml/table/vqvae.yml \
--geovae_ckpt_dir ./20210601_table_geo \
--geovae_yaml ./code_TMNET/yaml/table/geovae.yml \
--category table \
--save_path ./20210507_table_latents \
--device 3 \
--mode 'train'/'test'/'val'

# train top and bottom level pixelsnail for other parts
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/leg/pixelsnail_top.yml
python ./code_TMNET/train.py --yaml ./code_TMNET/yaml/table/leg/pixelsnail_bottom.yml

# sample textures for the seed part
python code_TMNET/conditional_sample_2levels_central_part.py \
--path ./20210601_table_latents \
--part_name surface \
--vqvae ./20210601_table/latest1.pth \
--vqvae_yaml ./code_TMNET/yaml/table/vqvae.yml \
--top ./20210601_table_pixelsnail/top_16/latest.pth \
--top_yaml ./code_TMNET/yaml/table/pixelsnail_top_center_16.yml \
--bottom ./20210601_table_pixelsnail/bottom/latest.pth \
--bottom_yaml ./code_TMNET/yaml/table/pixelsnail_bottom_center.yml \
--device 0 \
--batch 1

# sample textures for other parts
python code_TMNET/conditional_sample_2levels_other_parts.py \
--path ../../../205_f/wutong/latent_data/20210516_table_latents_others \
--central_part_name surface \
--part_name leg \
--vqvae ./20210601_table/latest1.pth \
--vqvae_yaml ./code_TMNET/yaml/table/vqvae.yml \
--top ./20210601_table_pixelsnail/leg/top_16/latest.pth \
--top_yaml ./code_TMNET/yaml/table/leg/pixelsnail_top_center_16.yml \
--bottom ./20210601_table_pixelsnail/leg/bottom/latest.pth \
--bottom_yaml ./code_TMNET/yaml/table/leg/pixelsnail_bottom_center.yml \
--central_part_sample_dir ./20210601_table_pixelsnail/top_16/auto_texture \
--device 0 \
--batch 1