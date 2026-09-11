#include "physics/hyperbolic_flux.hpp"
#include "physics/pressure_law.hpp"
#include "numerics/minmod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {
struct Reconstructed { double h,q,r; };
struct CharData { double lm,lp,maxabs,disc; };

CharData characteristics(double h, double q, const Parameters& p, const ClosurePoint& c) {
    const double hsafe=std::max(h,1e-12);
    // F2 = q^2/h * phi_sf(h,q) + P(h)
    const double a = -q*q*c.phi_sf/(hsafe*hsafe)
                   + q*q*c.dphi_sf_dh/hsafe
                   + LavallePressureLaw::dPdh(hsafe,p);
    const double b = 2.0*q*c.phi_sf/hsafe + q*q*c.dphi_sf_dq/hsafe;
    const double disc = b*b + 4.0*a;
    if (disc < 0.0) return {0.5*b,0.5*b,std::abs(0.5*b),disc};
    const double root=std::sqrt(disc);
    const double lm=0.5*(b-root), lp=0.5*(b+root);
    return {lm,lp,std::max(std::abs(lm),std::abs(lp)),disc};
}
}

NumericalFlux compute_rusanov_fluxes(const State& s, int ng, double,
                                     const Parameters& p, const ClosureModel& closure,
                                     FluxDiagnostics* diagnostics) {
    const int nx=static_cast<int>(s.size());
    NumericalFlux nf(static_cast<std::size_t>(nx-1));
    std::vector<std::array<double,3>> slope(static_cast<std::size_t>(nx), {0.0,0.0,0.0});
    for(int i=1;i<nx-1;++i) {
        slope[i][0]=minmod(s.h[i]-s.h[i-1], s.h[i+1]-s.h[i]);
        slope[i][1]=minmod(s.q[i]-s.q[i-1], s.q[i+1]-s.q[i]);
        slope[i][2]=minmod(s.r[i]-s.r[i-1], s.r[i+1]-s.r[i]);
    }

    // C++ face index j is the interface between cells j and j+1.
    for(int j=ng-1; j<=nx-ng-1; ++j) {
        Reconstructed L{s.h[j]+0.5*slope[j][0], s.q[j]+0.5*slope[j][1], s.r[j]+0.5*slope[j][2]};
        Reconstructed R{s.h[j+1]-0.5*slope[j+1][0], s.q[j+1]-0.5*slope[j+1][1], s.r[j+1]-0.5*slope[j+1][2]};
        L.h=std::max(L.h,1e-12); R.h=std::max(R.h,1e-12);

        const auto cL=closure.evaluate(L.h,L.q,p), cR=closure.evaluate(R.h,R.q,p);
        const auto chL=characteristics(L.h,L.q,p,cL), chR=characteristics(R.h,R.q,p,cR);
        if (chL.disc < 0.0 || chR.disc < 0.0)
            throw std::runtime_error("Loss of hyperbolicity: learned/analytical flux discriminant < 0");
        const double wavespeed=std::max(chL.maxabs,chR.maxabs);

        const double wL=L.r/L.h, wR=R.r/R.h;
        const std::array<double,3> FL{
            L.q,
            L.q*L.q/L.h*cL.phi_sf + LavallePressureLaw::P(L.h,p),
            L.q*wL
        };
        const std::array<double,3> FR{
            R.q,
            R.q*R.q/R.h*cR.phi_sf + LavallePressureLaw::P(R.h,p),
            R.q*wR
        };
        const std::array<double,3> dU{R.h-L.h,R.q-L.q,R.r-L.r};
        nf.h[j]=0.5*(FL[0]+FR[0])-0.5*wavespeed*dU[0];
        nf.q[j]=0.5*(FL[1]+FR[1])-0.5*wavespeed*dU[1];
        nf.r[j]=0.5*(FL[2]+FR[2])-0.5*wavespeed*dU[2];
        if(diagnostics){
            diagnostics->min_discriminant=std::min({diagnostics->min_discriminant,chL.disc,chR.disc});
            diagnostics->max_abs_lambda=std::max({diagnostics->max_abs_lambda,chL.maxabs,chR.maxabs});
        }
    }
    return nf;
}
