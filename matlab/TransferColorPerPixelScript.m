function TransferColorPerPixelScript(category,register_root,shapenet_root,save_dir)
% This function is used to generate texture images for registered parts from
% the origin model of ShapeNet.
% Input:
%     register_root: path to registered objs
%     shapenet_root: path to ShapeNet models
%     registered_unwrapper_obj_path: path to unwrapper box obj
%     save_dir: path to save texture image

part_names = getlabel(category);
if strcmp(category,'chair') || strcmp(category,'table')
    % alpha will help chair and table get better visual performance
    alpha_open = 1;
else
    alpha_open = 0;
end
registered_unwrapper_obj_path = fullfile(['.\cube_',category], 'cube_std_2d.obj');
if ~exist(registered_unwrapper_obj_path,'file')
    error('No unwrapper cube mesh cube_std_2d.obj!')
end

id_list = dir(register_root);
id_list = {id_list(:).name};
id_list(ismember(id_list,{'.','..'})) = [];
for i = 1:length(id_list)
    model_id = id_list{i};
    shapenet_path = fullfile(shapenet_root, model_id, 'models', 'model_normalized.obj');
    mtl_path = fullfile(shapenet_root, model_id, 'models', 'model_normalized.mtl');
    if ~exist(mtl_path,'file')
        mtl_path = fullfile(shapenet_root, model_id, 'model.mtl');
    end
    model_save_dir = fullfile(save_dir,model_id);
    if ~exist(model_save_dir,'dir')
        mkdir(model_save_dir)
    end

    for j = 1:size(part_names, 2)
        part_name = part_names{j};
        registered_obj_path = fullfile(register_root, model_id, [part_name, '_reg.obj']);
        if ~exist(registered_obj_path, 'file')
            continue;
        end
        save_name = fullfile(model_save_dir, [model_id, '_', part_name, '.png']);
        
        if strcmp(category,'car')
            if strcmp(part_name,'body')
                cmd = ['ray_tracing_4carbody.exe ', shapenet_path, ' ', mtl_path, ' ', registered_obj_path, ' ', registered_unwrapper_obj_path, ' ', save_name];
            else
                cmd = ['ray_tracing_4car.exe ', shapenet_path, ' ', mtl_path, ' ', registered_obj_path, ' ', registered_unwrapper_obj_path, ' ', save_name, ' 3'];
            end
        else
            cmd = ['ray_tracing.exe ', shapenet_path, ' ', mtl_path, ' ', registered_obj_path, ' ', registered_unwrapper_obj_path, ' ', save_name];
        end
        if ~exist(save_name,'file')
            system(cmd);
        end
        
        if alpha_open
            alpha_save_name = fullfile(model_save_dir, [model_id, '_', part_name, '_alpha.png']);
            cmd = ['ray_tracing_transparency.exe ', shapenet_path, ' ', mtl_path, ' ', registered_obj_path, ' ', registered_unwrapper_obj_path, ' ', alpha_save_name];
            system(cmd);
            origin_img = imread(save_name);
            alpha_img = imread(alpha_save_name);
            alpha = alpha_img(:,:,2);
            imwrite(origin_img, save_name, 'Alpha', alpha);
            delete(alpha_save_name);
        end
    end
    disp([model_id, ' texture finish!']);
end