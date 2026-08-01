clear; clc; close all;
clear chang_noise_signal append_thinfilm_h5; %% To clear the persistent variables and reinitialize the noise_phase


%% Read input
fileID = fopen('input.txt','r');
C = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\n');

fcut_dim  = C{1}
t_end_dim = C{2}
Ldim      = C{3}
N         = C{4}
MNOISE    = C{5}
F0_noise  = C{6}
Re_input  = C{7}
theta     = C{8}
rho       = C{9}
mu        = C{10}
gamma     = C{11}
t_save    = C{12}
fclose(fileID);

MNOISE = round(MNOISE);

%% Default values
% fcut_dim  = 14;
% t_end_dim = 0.03;
% Ldim      = 0.4;
% N         = 500;
% MNOISE    = 1000;
% F0_noise  = 0.025;
% Re_input  = 19.3;
% theta     = 6.4;
% rho       = 1072;
% mu        = 0.00673;
% gamma     = 6.7e-02;
% t_save = 0.01;
% theta = theta*pi/180;

%% Physical parameters
g = 9.81;
nu = mu/rho;

%% Dimensional scales
hN = (3*Re_input*nu^2/(g*sin(theta)))^(1/3);
uN = nu*Re_input/hN;
TN = hN/uN;

Re = uN*hN/nu;
F2 = uN^2/(g*hN);
We = rho*hN*uN^2/gamma;

fcut_noise = fcut_dim*TN;

fprintf('hN = %.4e m\n', hN);
fprintf('uN = %.4e m/s\n', uN);
fprintf('TN = %.4e s\n', TN);
fprintf('F2 = %.4f, We = %.4f, Re = %.4f, Ca = %.4f\n', F2, We, Re, We/Re);
fprintf('fcut_noise nondim = %.4e\n', fcut_noise);

%% Domain
L = Ldim/hN;
ng = 1;
Nx = N + 2*ng;

dx = L/N;
x = ((1:N) - 0.5)*dx;
x_dim = x*hN;

%% Time
t_end = t_end_dim/TN;
dt_dim = 5e-5;
dt = dt_dim/TN;

nsteps = ceil(t_end/dt);
dt = t_end/nsteps;
dt_dim_actual = dt*TN;

fprintf('dx = %.4e, dt = %.4e, nsteps = %d\n', dx, dt, nsteps);

%% Initial condition
h = ones(Nx,1);
q = ones(Nx,1);
r = zeros(Nx,1);

plot_every = max(1, floor(nsteps/250));
save_every = max(1, round(t_save/dt_dim_actual));

% case_name = sprintf('whitenoise_fcut_%gHz_F0_%g', fcut_dim, F0_noise);
%% Storage for inlet signal
t_nd_signal = zeros(nsteps,1);
t_dim_signal = zeros(nsteps,1);
F_signal = zeros(nsteps,1);
q_inlet_signal = zeros(nsteps,1);

figure;

%% Time loop
for n = 1:nsteps

    h_old = h;
    q_old = q;

    t = (n-1)*dt;


    %% Store inlet signal at physical time step
    F_now = chang_noise_signal(t, MNOISE, fcut_noise, F0_noise);
    t_nd_signal(n) = t;
    t_dim_signal(n) = t*TN;
    F_signal(n) = F_now;
    q_inlet_signal(n) = 1.0 + F_now;


    %% RK4 stage 1
    [k1h,k1q,k1r] = solveRHS( ...
        h,q,r,dx,ng,Re,F2,We,theta, ...
        MNOISE,fcut_noise,F0_noise,t);

    h2 = h + 0.5*dt*k1h;
    q2 = q + 0.5*dt*k1q;
    r2 = r + 0.5*dt*k1r;

    h2 = max(h2,1e-7);

    %% RK4 stage 2
    [k2h,k2q,k2r] = solveRHS( ...
        h2,q2,r2,dx,ng,Re,F2,We,theta, ...
        MNOISE,fcut_noise,F0_noise,t + 0.5*dt);

    h3 = h + 0.5*dt*k2h;
    q3 = q + 0.5*dt*k2q;
    r3 = r + 0.5*dt*k2r;

    h3 = max(h3,1e-7);

    %% RK4 stage 3
    [k3h,k3q,k3r] = solveRHS( ...
        h3,q3,r3,dx,ng,Re,F2,We,theta, ...
        MNOISE,fcut_noise,F0_noise,t + 0.5*dt);

    h4 = h + dt*k3h;
    q4 = q + dt*k3q;
    r4 = r + dt*k3r;

    h4 = max(h4,1e-7);

    %% RK4 stage 4
    [k4h,k4q,k4r] = solveRHS( ...
        h4,q4,r4,dx,ng,Re,F2,We,theta, ...
        MNOISE,fcut_noise,F0_noise,t + dt);

    %% Final RK4 update
    h = h + (dt/6)*(k1h + 2*k2h + 2*k3h + k4h);
    q = q + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q);
    r = r + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r);

    h = max(h,1e-7);

    %% Plot
    if mod(n,plot_every)==0 || n==1
        hi = h(ng+1:ng+N);

        plot(x_dim, hi, 'k-', 'LineWidth', 1.2);
        xlabel('$x$ (m)','interpreter','latex');
        ylabel('$h/h_0$','interpreter','latex');
        title(sprintf('Lavalle Solver: t = %.2f s', t*TN));
        grid on;
        ylim([0.6, 2]);
        drawnow;
    end

    %% Save snapshots
    if mod(n,save_every)==0

        hi = h(ng+1:ng+N);
        qi = q(ng+1:ng+N);
        ri = r(ng+1:ng+N);
        
        dhdt = (h - h_old)/dt;
        dqdt = (q - q_old)/dt;

        dhdt = dhdt(ng+1:ng+N);
        dqdt = dqdt(ng+1:ng+N);

        F_now = chang_noise_signal(t+dt, MNOISE, fcut_noise, F0_noise);
        q_inlet_now = 1.0 + F_now;
    
        params.Re = Re;
        params.We = We;
        params.F2 = F2;
        params.hN = hN;
        params.uN = uN;
        params.TN = TN;
        params.theta = theta;
        params.fcut_dim = fcut_dim;
        params.fcut_noise = fcut_noise;
        params.F0_noise = F0_noise;
        params.MNOISE = MNOISE;
    
        append_thinfilm_h5('thinfilm_training_data.h5', ...
            x, t, hi, qi, ri, dhdt, dqdt, F_now, q_inlet_now, params);
    end
end

%% Save inlet signal
save('inlet_signal.mat', ...
     't_dim_signal', 't_nd_signal', 'F_signal', 'q_inlet_signal', ...
     'fcut_dim', 'fcut_noise', 'F0_noise', 'MNOISE', 'TN');


%% Final fields
hi = h(ng+1:ng+N);
qi = q(ng+1:ng+N);
ri = r(ng+1:ng+N);

ui = qi./hi;
wi = ri./hi;

%% Final plots
close all;

figure(1);
plot(x_dim, hi, 'LineWidth', 3);
xlabel('$x$ (m)','interpreter','latex');
ylabel('$h/h_0$','interpreter','latex');
title(sprintf('$t = %.4f$ s', t_end_dim),'interpreter','latex');
grid on;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], ...
    'PaperUnits', 'Inches', 'PaperSize', [12 6]);
set(gca,'FontSize',18);
set(gca,'linewidth',3);
set(gca, 'FontName', 'Times');
name = 'interfaceheight.png';
print('-dpng','-r300',name);

figure;
plot(x_dim, uN*ui, 'LineWidth', 2);
xlabel('$x$ (m)','interpreter','latex');
ylabel('$u$ (m/s)','interpreter','latex');
title('Depth-averaged velocity');
grid on;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], ...
    'PaperUnits', 'Inches', 'PaperSize', [12 6]);
set(gca,'FontSize',18);
set(gca,'linewidth',3);
set(gca, 'FontName', 'Times');
name = 'meanvelocity.png';
print('-dpng','-r300',name);

figure;
plot(x_dim, wi, 'LineWidth', 2);
xlabel('$x$ (m)','interpreter','latex');
ylabel('$w$','interpreter','latex');
title('Augmented variable - Surface tension');
grid on;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 6], ...
    'PaperUnits', 'Inches', 'PaperSize', [12 6]);
set(gca,'FontSize',18);
set(gca,'linewidth',3);
set(gca, 'FontName', 'Times');
name = 'surfacetensionvariable_.png';
print('-dpng','-r300',name);

fname = 'Results.mat';
save(fname, 'x_dim', 'uN', 'hN', 'hi','ui','wi','Re','We','F2');