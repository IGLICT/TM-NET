function PrepareForTraining(category, vae_dir, final_dir, train_percent)
% generate and move data for training and test

if nargin < 4
    train_percent = 0.75;
end

part_names = getlabel(category);
id_list = dir(vae_dir);
dir_flag = [id_list.isdir];
id_list = {id_list(dir_flag).name};
id_list(ismember(id_list,{'.','..'})) = [];
data_num = length(id_list);
train_num = floor(data_num * train_percent);
rand_nums = randperm(data_num);
H_begin = [257, 1, 257, 513, 769, 257];
W_begin = [1, 257, 257, 257, 257, 513];
for i = 1:data_num
    data_id = rand_nums(i);
    model_id = id_list{data_id};
    if i <= train_num
        model_output = fullfile(final_dir, 'train', model_id);
    else
        model_output = fullfile(final_dir, 'test', model_id);
    end
    if ~exist(model_output,'dir')
        mkdir(model_output);
    end
    copyfile(fullfile(vae_dir, model_id, 'code.mat'),fullfile(model_output,'code.mat'))
    for j = 1:length(part_names)
        mat_file = fullfile(vae_dir, model_id, [model_id,'_',part_names{j},'.mat']);
        png_file = fullfile(vae_dir, model_id, [model_id,'_',part_names{j},'.png']);
        if exist(mat_file,'file')
            copyfile(mat_file,fullfile(model_output,[model_id,'_',part_names{j},'.mat']))
            if exist(png_file,'file')
                [img,~,alpha] = imread(png_file);
                img = imresize(img,[768,1024]);
                if isempty(alpha)
                    imwrite(img,fullfile(model_output,[model_id,'_',part_names{j},'.png']))
                else
                    alpha = imresize(alpha,[768,1024]);
                    imwrite(img,fullfile(model_output,[model_id,'_',part_names{j},'.png']), 'Alpha', alpha)
                end
                for m = 1:6
                    patch_file = fullfile(model_output,[model_id,'_',part_names{j},'_patch',num2str(m),'.png']);
                    patch_img = img(W_begin(m):W_begin(m)+255, H_begin(m):H_begin(m)+255, :);
                    if isempty(alpha)
                        imwrite(patch_img, patch_file);
                    else
                        patch_alpha = alpha(W_begin(m):W_begin(m)+255, H_begin(m):H_begin(m)+255);
                        imwrite(patch_img, patch_file, 'Alpha', patch_alpha);
                    end
                end
            else
                warning([model_id,'_',part_names{j},'.png loss!'])
            end
        end
    end
end
    
end