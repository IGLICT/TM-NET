function GenerateData(box_dir, acap_output_dir, category)
    part_names = getlabel(category);
    d = dir(box_dir);
    isub = [d(:).isdir];
    nameFolds = {d(isub).name}';
    nameFolds(ismember(nameFolds,{'.','..'})) = [];
    
    ref_mesh = fullfile(['.\cube_',category], 'cube_std.obj');
    if ~exist(ref_mesh,'file')
        error('No cube mesh!')
    end
    if ~exist(acap_output_dir, 'dir')
        mkdir(acap_output_dir);
    end
    
    for i = 1:size(nameFolds, 1)
        id = nameFolds{i};
        output_dir = fullfile(acap_output_dir, id);
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        if exist(fullfile(box_dir,id,'code.bad'),'file')
            copyfile(fullfile(box_dir, id, 'code.bad'),fullfile(output_dir,'code.mat'))
        elseif exist(fullfile(box_dir,id,'code.mat'),'file')
            copyfile(fullfile(box_dir, id, 'code.mat'),fullfile(output_dir,'code.mat'))
        end
        
        for j = 1:size(part_names, 2)
            deformed_mesh = fullfile(box_dir, id, [part_names{j}, '_reg.obj']);
            if exist(deformed_mesh, 'file')
                part_output_dir = fullfile(output_dir, part_names{j});
                if ~exist(part_output_dir, 'dir')
                    mkdir(part_output_dir);
                end
                mat_file = fullfile(output_dir, [id, '_', part_names{j}, '.mat']);
                if ~exist(fullfile(part_output_dir, 'S.txt'), 'file') || ~exist(fullfile(part_output_dir, 'LOGRNEW.txt'), 'file')
                    copyfile(ref_mesh, fullfile(part_output_dir, '0.obj'));
                    copyfile(deformed_mesh, part_output_dir);
                    try
                        ACAPOpt(part_output_dir);
                    catch
                        fprint('error id %s, %s', id, part_names{j});
                        continue;
                    end
                    % ACAPOpt2Meshes(ref_mesh, deformed_mesh, part_output_dir);
                end
                if exist(fullfile(part_output_dir, 'LOGRNEW.txt'), 'file') && exist(fullfile(part_output_dir, 'S.txt'), 'file') && ~exist(mat_file,'file')
                    LOGRNEW = dlmread(fullfile(part_output_dir, 'LOGRNEW.txt'));
                    S = dlmread(fullfile(part_output_dir, 'S.txt'));
                    LOGRNEW = LOGRNEW(2, :);
                    S = S(2, :);
                    pointnum = size(S, 2)/9;
                    
                    [ fmlogdr, fms ] = FeatureMap( LOGRNEW, S );
                    fmlogdr = permute(reshape(fmlogdr,size(fmlogdr,1),3,pointnum),[1,3,2]);
                    fms = permute(reshape(fms,size(fms,1),6,pointnum),[1,3,2]);
                    %             acap_feature = cat(3, fmlogdr, fms);
                    save(mat_file, 'fmlogdr', 'fms');
                end
            end
        end
        disp([id, ' ACAP finish!']);
    end

end
    
