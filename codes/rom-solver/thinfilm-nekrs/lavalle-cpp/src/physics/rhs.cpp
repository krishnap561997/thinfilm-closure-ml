#include "physics/rhs.hpp"
#include "physics/boundary_conditions.hpp"
#include "physics/capillary.hpp"
#include "physics/source_terms.hpp"

RHS evaluate_rhs(const State& input, double t, const Parameters& p,
                 const ClosureModel& closure, const ChangNoise& noise,
                 FluxDiagnostics* diagnostics) {
    State s=input; // MATLAB solveRHS applies BC to local copies.
    apply_white_noise_bc(s,p.ng,noise,t);

    auto flux=compute_rusanov_fluxes(s,p.ng,p.dx,p,closure,diagnostics);
    RHS rhs(s.size());
    const int nx=static_cast<int>(s.size());
    for(int i=p.ng;i<nx-p.ng;++i){
        rhs.h[i]=-(flux.h[i]-flux.h[i-1])/p.dx;
        rhs.q[i]=-(flux.q[i]-flux.q[i-1])/p.dx;
        rhs.r[i]=-(flux.r[i]-flux.r[i-1])/p.dx;
    }

    const auto cap=compute_augmented_capillary(s,p);
    const auto src=compute_momentum_source(s,p,closure);
    const auto visc=compute_viscous_diffusion(s,p);
    for(int i=p.ng;i<nx-p.ng;++i){
        rhs.q[i]+=src[i]+visc[i]+cap.q[i];
        rhs.r[i]+=cap.r[i];
    }
    // Ghost RHS stays zero by construction.
    return rhs;
}
