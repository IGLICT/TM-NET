function ViewOBJandTexture(obj_dir, texture_dir, cube_std_obj, model_id)
    [V1,F1,UV1,TF1,N1,NF1] = readOBJ(cube_std_obj);
    filenames = dir(fullfile(obj_dir,'*_reg.obj'));
    for k = 1:numel(filenames)
        filename = fullfile(filenames(k).folder, filenames(k).name);
        strparts = strsplit(filenames(k).name, '.');
        name = strparts{1};
        strparts = strsplit(name, '_');
        part_name = strparts{1};
        for j = 2:size(strparts, 2)-1
            part_name = [part_name, '_', strparts{j}];
        end
        
        src_filename = fullfile(texture_dir, [model_id, '_', part_name, '.png']);
        tar_filename = fullfile(obj_dir, [part_name, '_reg.png']);
        if exist(src_filename, 'file')
            copyfile(src_filename, tar_filename);
            [V,F,UV,TF,N,NF] = readOBJ(filename);
            WriteOBJwithMtl(filename, V,F,UV1,TF1,N1,NF1);
            WriteMtl(fullfile(obj_dir, [part_name, '.mtl']));
        end
    end
end