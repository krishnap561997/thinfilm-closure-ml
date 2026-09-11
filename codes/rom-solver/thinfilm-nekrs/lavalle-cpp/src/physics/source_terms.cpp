#include "physics/source_terms.hpp"
#include "physics/pressure_law.hpp"
#include <algorithm>

Field compute_momentum_source(const State& s, const Parameters& p, const ClosureModel& closure) {
    Field out(s.size(),0.0); const double L=LavallePressureLaw::lambda(p);
    for(std::size_t i=0;i<s.size();++i){
        const double h=std::max(s.h[i],1e-12);
        const auto c=closure.evaluate(h,s.q[i],p);
        out[i]=(L*h-c.tau_w)/p.Re;
    }
    return out;
}
Field compute_viscous_diffusion(const State& s, const Parameters& p) {
    Field out(s.size(),0.0); const int nx=static_cast<int>(s.size());
    for(int i=p.ng;i<nx-p.ng;++i)
        out[i]=(2.0/p.Re)*(s.q[i+1]-2.0*s.q[i]+s.q[i-1])/(p.dx*p.dx);
    return out;
}
