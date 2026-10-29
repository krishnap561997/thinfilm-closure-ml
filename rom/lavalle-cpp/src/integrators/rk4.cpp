#include "integrators/rk4.hpp"
#include "physics/rhs.hpp"
#include <algorithm>

namespace {
State axpy(const State& base, const RHS& k, double a, double floor_h){
    State out=base;
    for(std::size_t i=0;i<out.size();++i){
        out.h[i]=std::max(base.h[i]+a*k.h[i],floor_h);
        out.q[i]=base.q[i]+a*k.q[i];
        out.r[i]=base.r[i]+a*k.r[i];
    }
    return out;
}
}

void rk4_step(State& s, double t, double dt, const Parameters& p,
              const ClosureModel& closure, const ChangNoise& noise){
    const RHS k1=evaluate_rhs(s,t,p,closure,noise);
    const State s2=axpy(s,k1,0.5*dt,p.h_floor);
    const RHS k2=evaluate_rhs(s2,t+0.5*dt,p,closure,noise);
    const State s3=axpy(s,k2,0.5*dt,p.h_floor);
    const RHS k3=evaluate_rhs(s3,t+0.5*dt,p,closure,noise);
    const State s4=axpy(s,k3,dt,p.h_floor);
    const RHS k4=evaluate_rhs(s4,t+dt,p,closure,noise);

    for(std::size_t i=0;i<s.size();++i){
        s.h[i]=std::max(s.h[i]+dt*(k1.h[i]+2*k2.h[i]+2*k3.h[i]+k4.h[i])/6.0,p.h_floor);
        s.q[i]+=dt*(k1.q[i]+2*k2.q[i]+2*k3.q[i]+k4.q[i])/6.0;
        s.r[i]+=dt*(k1.r[i]+2*k2.r[i]+2*k3.r[i]+k4.r[i])/6.0;
    }
}
