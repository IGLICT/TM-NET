function MergeOBJWithTexture(obj_dir, category, model_id)
    if nargin < 3
        model_id = '';
    end
    labels = getlabel(category);
    if strcmp(category,'chair') || strcmp(category,'table')
        alpha_open = 1;
    else
        alpha_open = 0;
    end
    
    d = dir(obj_dir);
    isub = [d(:).isdir];
    nameFolds = {d(isub).name}';
    nameFolds(ismember(nameFolds,{'.','..'})) = [];
    
    parfor i = 1:size(nameFolds, 1)
        id = nameFolds{i};
        if ~strcmp(model_id,'') && ~strcmp(id, model_id)
            continue;
        end
        disp(id);
        if exist(fullfile(obj_dir, id, 'merge.obj'), 'file')
            continue;
        end
        merge_V = [];
        merge_F = [];
        merge_UV = [];
        merge_TF = [];
        merge_texture = [];
        if alpha_open
            merge_alpha = [];
        end
        true_part_num = 0;
        for j = 1:size(labels, 2)
            part_name = fullfile(obj_dir, id, [labels{j}, '_reg.obj']);
            if exist(part_name, 'file')
                true_part_num = true_part_num + 1;
            end
        end
        count = 1;
        for j = 1:size(labels, 2)
            part_name = fullfile(obj_dir, id, [labels{j}, '_reg.obj']);
            png_name = fullfile(obj_dir, id, [labels{j}, '_reg.png']);
            if exist(part_name, 'file') && exist(png_name, 'file')
                [V, F, UV, TF] = readOBJ(part_name);
                [texture,~,alpha] = imread(png_name);
            else
                continue;
            end
            
            F = F + size(merge_V, 1);
            merge_F = [merge_F; F];
            
            TF = TF + size(merge_UV, 1);
            merge_TF = [merge_TF; TF];
            
            UV(:, 1) = 1.0*(count-1)/true_part_num + 1.0*UV(:, 1)/true_part_num;
            merge_UV = [merge_UV; UV];
            
            merge_V = [merge_V; V];
            
            if isempty(merge_texture)
                merge_texture = texture;
            else
                merge_texture = cat(2,merge_texture,texture);
            end
            if alpha_open
                if isempty(merge_alpha)
                    merge_alpha = alpha;
                else
                    merge_alpha = cat(2,merge_alpha,alpha);
                end
            end
            count = count + 1;
        end
        merge_name = fullfile(obj_dir, id, 'merge.obj');
        mtl_name = fullfile(obj_dir, id, 'merge.mtl');
        png_name = fullfile(obj_dir, id, 'merge.png');
        if ~isempty(merge_V)
            WriteOBJwithMtl(merge_name, merge_V, merge_F, merge_UV, merge_TF);
            WriteMtl(mtl_name);
            if alpha_open
                imwrite(merge_texture, png_name, 'Alpha', merge_alpha);
            else
                imwrite(merge_texture, png_name);
            end
        end
    end
end