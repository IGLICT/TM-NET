% Prepare data:
% 1. divide model to multiple parts
% 2. get transformed cube
% 3. regist cube to part
% 4. get structure code
% 5. generate deformation feature
% 6. generate texture image

% Chair
addpath('.\nonregistration')
cate = 'chair';
postfix = 50;
data_dir = '.\DATA\Chair\Chair';
shapenet_root = '.\ShapeNet\Chair';
box_dir = fullfile(data_dir,'..',['box',num2str(postfix)]);
vae_dir = fullfile(data_dir,'..',['vaenew',num2str(postfix)]);
final_dir = fullfile(data_dir,'..',['final',num2str(postfix)]);

GetTransformedCube(data_dir, postfix, cate);
regist(box_dir, cate)
SupportAnalysisScript(box_dir, cate);
GenerateData(box_dir, vae_dir, cate);
TransferColorPerPixelScript(cate, box_dir, shapenet_root, vae_dir)
PrepareForTraining(cate, vae_dir, final_dir, 0.75);

% next steps:
% 1. train TM-NET
% 2. generate single textured part by 'ViewOBJandTexture.m'
% 3. merge parts to the whole textured model by 'MergeOBJWithTexture.m'
