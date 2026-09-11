#include "numerics/derivatives.hpp"
#include <stdexcept>

Field ddx_1d(const Field& f, double dx) {
    const std::size_t n=f.size(); if(n<3) throw std::runtime_error("ddx needs >=3 points");
    Field out(n,0.0);
    for(std::size_t i=1;i+1<n;++i) out[i]=(f[i+1]-f[i-1])/(2.0*dx);
    out[0]=(-3.0*f[0]+4.0*f[1]-f[2])/(2.0*dx);
    out[n-1]=(3.0*f[n-1]-4.0*f[n-2]+f[n-3])/(2.0*dx);
    return out;
}
Field d2dx2_1d(const Field& f, double dx) {
    const std::size_t n=f.size(); if(n<4) throw std::runtime_error("d2dx2 needs >=4 points");
    Field out(n,0.0); const double d=dx*dx;
    for(std::size_t i=1;i+1<n;++i) out[i]=(f[i+1]-2.0*f[i]+f[i-1])/d;
    out[0]=(2.0*f[0]-5.0*f[1]+4.0*f[2]-f[3])/d;
    out[n-1]=(2.0*f[n-1]-5.0*f[n-2]+4.0*f[n-3]-f[n-4])/d;
    return out;
}
Field d3dx3_1d(const Field& f, double dx) {
    const std::size_t n=f.size(); if(n<5) throw std::runtime_error("d3dx3 needs >=5 points");
    Field out(n,0.0); const double d=2.0*dx*dx*dx;
    for(std::size_t i=2;i+2<n;++i) out[i]=(f[i+2]-2.0*f[i+1]+2.0*f[i-1]-f[i-2])/d;
    out[0]=out[2]; out[1]=out[2]; out[n-2]=out[n-3]; out[n-1]=out[n-3];
    return out;
}
