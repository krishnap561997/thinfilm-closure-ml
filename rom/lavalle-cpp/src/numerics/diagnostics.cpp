#include "core/diagnostics.hpp"
#include "numerics/derivatives.hpp"
#include "physics/hyperbolic_flux.hpp"
#include <algorithm>
#include <cmath>

Diagnostics compute_diagnostics(const State& s, const Parameters& p, const ClosureModel& closure){
    Diagnostics d;
    const int n=p.in.N;
    Field hi(n);
    for(int i=0;i<n;++i) hi[i]=s.h[p.ng+i];
    const auto hx=ddx_1d(hi,p.dx);
    d.h_min=*std::min_element(hi.begin(),hi.end()); d.h_max=*std::max_element(hi.begin(),hi.end());
    for(double x:hx) d.max_abs_hx=std::max(d.max_abs_hx,std::abs(x));
    FluxDiagnostics fd; (void)compute_rusanov_fluxes(s,p.ng,p.dx,p,closure,&fd);
    d.min_hyperbolic_discriminant=fd.min_discriminant; d.max_abs_lambda=fd.max_abs_lambda;
    return d;
}
