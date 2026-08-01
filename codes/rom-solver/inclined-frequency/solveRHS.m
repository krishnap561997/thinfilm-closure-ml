%% ================================================================
%% RHS FOR AUGMENTED SYSTEM
%% ================================================================
function [rhs_h,rhs_q,rhs_r] = solveRHS(h,q,r,dx,ng,Re,F2,We,theta,...
                                             freq,F0,t)

    Nx = length(h);
    N = Nx - 2*ng;

    [h,q,r] = apply_BC(h,q,r,ng,freq,F0,t);

    u = q./h;
    w = r./h;

    %% Pressure law
    Lam = sin(theta)*Re/F2;

    flux_h = zeros(Nx-1,1);
    flux_q = zeros(Nx-1,1);
    flux_r = zeros(Nx-1,1);

    V = [h(:), q(:), r(:)];
    slope = zeros(size(V));

    %% MUSCL slopes
    for m = 1:3
        dL = V(2:end-1,m) - V(1:end-2,m);
        dR = V(3:end,m)   - V(2:end-1,m);

        slope(2:end-1,m) = minmod(dL,dR);
    end

    %% Rusanov fluxes
    for j = ng:Nx-ng
        VL = V(j,:)   + 0.5*slope(j,:);
        VR = V(j+1,:) - 0.5*slope(j+1,:);

        hL = max(VL(1),1e-12);
        qL = VL(2);
        rL = VL(3);

        hR = max(VR(1),1e-12);
        qR = VR(2);
        rR = VR(3);

        uL = qL/hL;
        uR = qR/hR;

        wL = rL/hL;
        wR = rR/hR;

        PL = cos(theta)*hL^2/(2*F2) + 2*Lam^2*hL^5/225;
        PR = cos(theta)*hR^2/(2*F2) + 2*Lam^2*hR^5/225;

        e2L = cos(theta)*hL/F2 + (2/45)*Lam^2*hL^4;
        e2R = cos(theta)*hR/F2 + (2/45)*Lam^2*hR^4;

        cL = sqrt(max(e2L,0));
        cR = sqrt(max(e2R,0));

        FL = [qL, qL^2/hL + PL, qL*wL];
        FR = [qR, qR^2/hR + PR, qR*wR];

        a = max(abs(uL)+cL, abs(uR)+cR);

        flux_rusanov = 0.5*(FL + FR) - 0.5*a*(VR - VL);

        flux_h(j) = flux_rusanov(1);
        flux_q(j) = flux_rusanov(2);
        flux_r(j) = flux_rusanov(3);
    end

    %% Conservative RHS
    rhs_h = zeros(Nx,1);
    rhs_q = zeros(Nx,1);
    rhs_r = zeros(Nx,1);

    for j = ng+1:ng+N
        rhs_h(j) = -(flux_h(j) - flux_h(j-1))/dx;
        rhs_q(j) = -(flux_q(j) - flux_q(j-1))/dx;
        rhs_r(j) = -(flux_r(j) - flux_r(j-1))/dx;
    end

    %% Augmented capillary terms
    sigma = 1/We;

    u = q./h;
    w = r./h;

    cap_q = zeros(Nx,1);
    cap_r = zeros(Nx,1);

    phi_face = zeros(Nx-1,1);
    wx_face = zeros(Nx-1,1);
    ux_face = zeros(Nx-1,1);

    for j = 1:Nx-1
        h_face = 0.5*(h(j) + h(j+1));

        phi_face(j) = sqrt(sigma)*h_face^(3/2);

        wx_face(j) = (w(j+1) - w(j))/dx;
        ux_face(j) = (u(j+1) - u(j))/dx;
    end

    for j = ng+1:Nx-ng
        cap_q(j) = ...
            (phi_face(j)*wx_face(j) ...
           - phi_face(j-1)*wx_face(j-1))/dx;

        cap_r(j) = ...
           -(phi_face(j)*ux_face(j) ...
           - phi_face(j-1)*ux_face(j-1))/dx;
    end

    %% Viscous diffusion
    qxx = zeros(Nx,1);

    for j = ng+1:ng+N
        qxx(j) = (q(j+1) - 2*q(j) + q(j-1))/dx^2;
    end

    viscous = 2*qxx/Re;

    %% Source / relaxation term
    source = (Lam*h - 3*u./h)/Re;

    rhs_q = rhs_q + source + viscous + cap_q;
    rhs_r = rhs_r + cap_r;

    %% Freeze ghost-cell RHS
    rhs_h(1:ng) = 0;
    rhs_q(1:ng) = 0;
    rhs_r(1:ng) = 0;

    rhs_h(end-ng+1:end) = 0;
    rhs_q(end-ng+1:end) = 0;
    rhs_r(end-ng+1:end) = 0;
end
