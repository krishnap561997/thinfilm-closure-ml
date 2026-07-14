clc; clear; close all;

%% Parameter values to sweep
Ca_list    = [0.001, 0.01, 0.1];
Re_list    = [5, 20, 40, 60, 80];
theta_list = [6.4, 45];   % degrees, unless your solver expects radians

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

%% Paths
parentFolder = '/blue/bala1s/krishnap.kalivel/ThinFilmFlow/thinfilm-closure-ml/rom-solver/Data_Ca_Re_sweep';

src_main = '/blue/bala1s/krishnap.kalivel/ThinFilmFlow/thinfilm-closure-ml/rom-solver/inclined-whitenoise/main_whitenoise';
src_run  = '/blue/bala1s/krishnap.kalivel/ThinFilmFlow/thinfilm-closure-ml/rom-solver/inclined-whitenoise/run.sh';

mkdir(parentFolder);

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
            mkdir(dir_path);

            %% Go to folder
            cd(dir_path);

            %% Write input.txt
            fid = fopen('input.txt', 'wt');

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

            fclose(fid);

            %% Copy executable/script files
            copyfile(src_main, 'main_whitenoise');
            copyfile(src_run,  'run.sh');

            %% Append commands to mscript
            fprintf(fid_mscript, 'cd %s\n', dir_path);
            fprintf(fid_mscript, 'sbatch run.sh\n');

        end
    end
end

fclose(fid_mscript);

cd(parentFolder);

fprintf('Generated %d cases.\n', ...
    length(Ca_list)*length(Re_list)*length(theta_list));

fprintf('Run all jobs using:\n');
fprintf('bash mscript\n');