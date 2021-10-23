function CalculateEdgeIndex(obj_file)
    [V, F, ~, ~, ~, VVsimp, CotWeight,~,~,~,edge_index] = cotlp(obj_file);
    edge_index=[edge_index;edge_index(:,[2,1])]';
    F = F';
    edge_index = edge_index';
    [filepath,name,ext] = fileparts(obj_file);
    save(fullfile(filepath, [name, '.mat']), 'V', 'F', 'edge_index');
end
