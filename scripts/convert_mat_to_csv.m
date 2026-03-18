% convert_mat_to_csv.m
% Converts CAMARGO .mat files (MATLAB tables) to CSV
% Usage: matlab -nodisplay -nosplash -r "run('convert_mat_to_csv.m')"

base = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'camargo', 'AB06', '10_09_18');
modes = {'levelground', 'ramp', 'stair'};
sensors = {'imu', 'gon', 'conditions'};

count = 0;
for m = 1:length(modes)
    for s = 1:length(sensors)
        folder = fullfile(base, modes{m}, sensors{s});
        if ~isfolder(folder), continue; end
        files = dir(fullfile(folder, '*.mat'));
        for f = 1:length(files)
            fp = fullfile(folder, files(f).name);
            d = load(fp);
            fn = fieldnames(d);
            t = d.(fn{1});
            if istable(t)
                out = strrep(fp, '.mat', '.csv');
                writetable(t, out);
                count = count + 1;
                fprintf('OK: %s (%dx%d)\n', out, size(t));
            else
                fprintf('SKIP (not table): %s\n', fp);
            end
        end
    end
end
fprintf('Converted %d files.\n', count);
exit;