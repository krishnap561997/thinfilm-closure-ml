%% ================================================================
%% BOUNDARY CONDITIONS - CHANG WHITE NOISE
%% ================================================================
function [h,q,r] = apply_BC(h,q,r,ng,f,F0,t)

    Nx = length(h);

    F_noise = inlet_signal(t, f, F0);
    inlet_factor = 1.0 + F_noise;

    %% Inlet
    for k = 1:ng
        q(k) = inlet_factor;
        h(k) = 1.0;
        r(k) = 0.0;
    end

    %% Outlet: zero-gradient
    for k = 0:ng-1
        h(Nx-k) = h(Nx-ng);
        q(Nx-k) = q(Nx-ng);
        r(Nx-k) = r(Nx-ng);
    end
end
