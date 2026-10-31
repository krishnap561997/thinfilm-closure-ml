#include "config/parameters.hpp"   // Reads input.txt and computed ND scales/params
#include "core/state.hpp"          // Holds evolving fields h, q and r
#include "core/diagnostics.hpp"    // Checks minimum h, maximum h_x, hyperbolicity
#include "closures/analytical_lavalle.hpp"  // Analytical closure relations
#include "forcing/chang_noise.hpp"          // Chang white noise inlet BC
#include "integrators/rk4.hpp"              // RK4 time stepper
#include "io/hdf5_writer.hpp"               // Output hdf5 files
#include "oracle/oracle_forcing.hpp"         // ORACLE
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

int main(int argc,char** argv){      //Arguments let you run either ./thinfilm or ./thinfilm my_input.txt [oracle.txt]
    try{
        const std::string input=(argc>1)?argv[1]:"input.txt"; //Default to input.txt if not my_input.txt not provided

        Parameters p=Parameters::from_file(input);


        std::cout
          << std::scientific
          << std::setprecision(6)
          << "\n--- Physical scales ---\n"
          << "hN          = " << p.hN << " m\n"
          << "uN          = " << p.uN << " m/s\n"
          << "TN          = " << p.TN << " s\n"
          << "\n--- Domain ---\n"
          << "L           = " << p.L << '\n'
          << "Ldim        = " << p.Ldim << " m\n"
          << "N           = " << p.in.N << '\n'
          << "dx          = " << p.dx << '\n'
          << "\n--- Time ---\n"
          << "t_end       = " << p.t_end << '\n'
          << "t_end_dim   = " << p.t_end_dim << " s\n"
          << "dt          = " << p.dt << '\n'
          << "dt_dim      = " << p.dt_dim << " s\n"
          << "nsteps      = " << p.nsteps << '\n'
          << "t_save      = " << p.t_save << '\n'
          << "t_save_dim  = " << p.t_save_dim << " s\n"
          << "\n--- Dimensionless groups ---\n"
          << "Re          = " << p.Re << '\n'
          << "Ca          = " << p.Ca << '\n'
          << "We          = " << p.We << '\n'
          << "F2          = " << p.F2 << '\n'
          << std::endl;

        State state(static_cast<std::size_t>(p.Nx));  //Create a solution state, State ( Fields h,q,r)

        // Initial condition
        std::fill(state.h.begin(),state.h.end(),1.0); std::fill(state.q.begin(),state.q.end(),1.0); 
        std::fill(state.r.begin(),state.r.end(),0.0);
        double t0=0.0;  // start time of the run

        std::unique_ptr<OracleForcing> oracle;  // ORACLE
        if(argc>2){ oracle=std::make_unique<OracleForcing>(argv[2],p); OracleForcing::set_active(oracle.get()); oracle->set_initial_state(state,p); t0=oracle->t_start(); std::cout<<oracle->summary(); }  // ORACLE

        // Inlet forcing. noise.value(t) gives forcing value at time t
        ChangNoise noise(p.in.MNOISE,p.fcut_noise,p.in.F0_noise,1);

        // Analytical Closure terms. We can change it to ML closure terms later. 
        AnalyticalLavalleClosure closure;


        // Create physical grid coordinates. Change it to a SEM mesh when NekRS
        Field x(static_cast<std::size_t>(p.in.N));
        for(int i=0;i<p.in.N;++i) x[i]=(i+0.5)*p.dx; //Cell-centered coordinates

        // Output writers - State variables, inlet signal
        H5TrainingWriter writer("thinfilm_training_data.h5",x,p);
        std::ofstream inlet("inlet_signal.csv"); inlet<<"t_nd,t_dim,F,q_inlet\n";

        // Start run-time solver. chrono cpp equivalent of tic toc
        auto start=std::chrono::steady_clock::now();
        for(int n=0;n<p.nsteps;++n){
            const double t=t0+n*p.dt;
            const State old=state;
            const double F0=noise.value(t);
            inlet<<std::setprecision(17)<<t<<','<<t*p.TN<<','<<F0<<','<<1.0+F0<<'\n';

            rk4_step(state,t,p.dt,p,closure,noise);  //RK4 step has the numerical pipeline
            // BC -> MUSCL recon -> closure -> Rusanov flux -> flux div -> source + viscous +
            // capillary

            // Checks diagnostics every progress_every timesteps
            if((n+1)%p.progress_every==0 || n==0){
                const auto d=compute_diagnostics(state,p,closure);
                std::cout<<"step "<<(n+1)<<" / "<<p.nsteps<<"; t="<<(t+p.dt)*p.TN<<" s"
                         <<"; h_min="<<d.h_min<<"; |hx|_max="<<d.max_abs_hx
                         <<"; min(discriminant)="<<d.min_hyperbolic_discriminant<<'\n';
            }

            // Saves h5 files every save_every timesteps
            if((n+1)%p.save_every==0){
                Field h(p.in.N),q(p.in.N),r(p.in.N),dhdt(p.in.N),dqdt(p.in.N);
                for(int i=0;i<p.in.N;++i){
                    const int I=p.ng+i;
                    h[i]=state.h[I]; q[i]=state.q[I]; r[i]=state.r[I];
                    dhdt[i]=(state.h[I]-old.h[I])/p.dt; dqdt[i]=(state.q[I]-old.q[I])/p.dt;
                }
                const double state_time=t+p.dt; 
                const double F=noise.value(state_time);
                writer.append(state_time,h,q,r,dhdt,dqdt,F,1.0+F);
            }
        }

        
        Field hf(p.in.N),qf(p.in.N),rf(p.in.N);
        for(int i=0;i<p.in.N;++i){hf[i]=state.h[p.ng+i];qf[i]=state.q[p.ng+i];rf[i]=state.r[p.ng+i];}
        std::ofstream final("final_state.csv"); final<<"x_dim,h,u_dim,w\n";
        for(int i=0;i<p.in.N;++i) final<<std::setprecision(17)<<x[i]*p.hN<<','<<hf[i]<<','<<p.uN*qf[i]/hf[i]<<','<<rf[i]/hf[i]<<'\n';

        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"Completed in "<<elapsed<<" s\n";
    }catch(const std::exception& e){ std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
    return 0;
}
