function WriteMtl(filename)
	[~, name, ~] = fileparts(filename);
	f = fopen(filename, 'w');
	format = ['newmtl material', newline, 'Kd 1 1 1', newline, 'Ka 0 0 0', newline, 'Ks 0.4 0.4 0.4', newline, 'Ke 0 0 0', newline, 'Ns 10', newline, 'illum 2', newline, 'map_Kd ./%s.png', newline];
    fprintf(f, format, name);
	fclose(f);
end
