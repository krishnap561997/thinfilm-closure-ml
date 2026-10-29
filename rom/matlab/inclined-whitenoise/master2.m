clc; clear; close all;

%% Parameter values to sweep
Ca_list    = [0.001];
Re_list    = [5, 20, 40, 60, 80];
theta_list = [6.4, 45];   % degrees

%% Fixed input parameters
fcut_dim  = 20;
t_end_dim = 15;
Ldim      = 2;
N         = 8000;
MNOISE    = 1000;
F0_noise  = 0.025;
rho       = 1072;
mu        = 0.00673;
t_save    = 0.01;
dt_dim    = 2e-5;

%% Paths

% Location where the simulation folders will be created
parentFolder = ...
    '/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/ROM_Ca_Re_sweep';

% Location containing the source executable and run script
sourceFolder = ...
    '/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/rom-solver/inclined-whitenoise';

src_main = fullfile(sourceFolder, 'main_whitenoise');
src_run  = fullfile(sourceFolder, 'run.sh');

%% Create parent folder
if ~exist(parentFolder, 'dir')
    mkdir(parentFolder);
end

%% Create mscript
mscript_path = fullfile(parentFolder, 'mscript');
fid_mscript = fopen(mscript_path, 'wt');

%% Loop over Ca, Re, theta
for iCa = 1:length(Ca_list)
    for iRe = 1:length(Re_list)
        for iTh = 1:length(theta_list)

            Ca_input = Ca_list(iCa);
            Re_input = Re_list(iRe);
            theta = theta_list(iTh);

            %% Folder name
            dirtext = sprintf('Ca_%07.4f_Re_%07.3f_theta_%06.3f', ...
                              Ca_input, Re_input, theta);

            dir_path = fullfile(parentFolder, dirtext);

            if ~exist(dir_path, 'dir')
                mkdir(dir_path);
            end

            %% Write input.txt directly into case folder
            input_path = fullfile(dir_path, 'input.txt');
            fid = fopen(input_path, 'wt');

            fprintf(fid, '%.12g\n', fcut_dim);
            fprintf(fid, '%.12g\n', t_end_dim);
            fprintf(fid, '%.12g\n', Ldim);
            fprintf(fid, '%.12g\n', N);
            fprintf(fid, '%.12g\n', MNOISE);
            fprintf(fid, '%.12g\n', F0_noise);
            fprintf(fid, '%.12g\n', Re_input);
            fprintf(fid, '%.12g\n', theta);
            fprintf(fid, '%.12g\n', rho);
            fprintf(fid, '%.12g\n', mu);
            fprintf(fid, '%.12g\n', Ca_input);
            fprintf(fid, '%.12g\n', t_save);
            fprintf(fid, '%.12g\n', dt_dim);

            fclose(fid);

            %% Copy executable/script files
            copyfile(src_main, fullfile(dir_path, 'main_whitenoise'));
            copyfile(src_run,  fullfile(dir_path, 'run.sh'));

            %% Append commands to mscript
            fprintf(fid_mscript, 'cd %s\n', dir_path);
            fprintf(fid_mscript, 'sbatch run.sh\n');

        end
    end
end

fclose(fid_mscript);

fprintf('Generated %d cases.\n', ...
    length(Ca_list)*length(Re_list)*length(theta_list));

fprintf('Cases created in:\n%s\n', parentFolder);

fprintf('Run all jobs using:\n');
fprintf('bash %s\n', mscript_path);
