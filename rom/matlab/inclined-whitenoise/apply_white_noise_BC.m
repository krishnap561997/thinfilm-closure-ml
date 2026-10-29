%% ================================================================
%% BOUNDARY CONDITIONS - CHANG WHITE NOISE
%% ================================================================
function [h,q,r] = apply_white_noise_BC(h,q,r,ng,MNOISE,fcut_noise,F0_noise,t)

    Nx = length(h);

    F_noise = chang_noise_signal(t, MNOISE, fcut_noise, F0_noise);
    inlet_factor = 1.0 + F_noise;

    %% Inlet
    for k = 1:ng
        q(k) = inlet_factor;
        h(k) = 1.0;
        r(k) = 0.0;
    end

    q(ng+1) = inlet_factor;
    h(ng+1) = 1.0;
    r(ng+1) = 0.0;

    %% Outlet: zero-gradient
    for k = 0:ng-1
        h(Nx-k) = h(Nx-ng);
        q(Nx-k) = q(Nx-ng);
        r(Nx-k) = r(Nx-ng);
    end
end