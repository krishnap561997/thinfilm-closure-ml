#include "physics/capillary.hpp"
#include <algorithm>
#include <cmath>

CapillaryRHS compute_augmented_capillary(const State& s, const Parameters& p) {
    const int nx=static_cast<int>(s.size());
    Field u(nx),w(nx);
    for(int i=0;i<nx;++i){ const double h=std::max(s.h[i],1e-12); u[i]=s.q[i]/h; w[i]=s.r[i]/h; }
    Field phi(nx-1),wx(nx-1),ux(nx-1);
    const double sigma=1.0/p.We;
    for(int j=0;j<nx-1;++j){
        const double hf=0.5*(s.h[j]+s.h[j+1]);
        phi[j]=std::sqrt(sigma)*std::pow(std::max(hf,1e-12),1.5);
        wx[j]=(w[j+1]-w[j])/p.dx;
        ux[j]=(u[j+1]-u[j])/p.dx;
    }
    CapillaryRHS out(s.size());
    for(int i=p.ng;i<nx-p.ng;++i){
        out.q[i]=(phi[i]*wx[i]-phi[i-1]*wx[i-1])/p.dx;
        out.r[i]=-(phi[i]*ux[i]-phi[i-1]*ux[i-1])/p.dx;
    }
    return out;
}
