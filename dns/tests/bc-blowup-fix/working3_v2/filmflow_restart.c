#include "grid/multigrid.h"
#include "navier-stokes/centered.h"

#include "two-phase-clsvof.h"
#include "integral.h"
#include "curvature.h"

/*#include "two-phase.h"
  #include "tension.h"*/

#include "navier-stokes/conserving.h"
//#include "tag.h"
//#include "reduced.h"
#include "view.h"
//vector h[];

#include "hdf5_headers/output_xdmf_params.h"

double t_out = 10;
double t_dump = 10;
double t_end = 100;
double RE = 20.0,
  CA = 0.01;
double GAMMA = 0.067;
double RHO_L = 1072,
  RHO_G = 1.225,
  MU_L = 0.00673,
  MU_G = 1.7894e-05;
double grav = 9.81;

double G[2];
double LX = 100;
int AR = 16, zoomy = 16;

int MAXlevel = 11;
//double uemax = 0.0001;

double angle = 6.4*pi/180.;
//double au = 0.03, freq = 1.5;
scalar f0[], profile[];

scalar f_minus[], p_minus[], d_minus[];
vector u_minus[];
double t_minus = 0.;

// White-band noise at inlet
int MNOISE = 1000;
double F0_noise = 0.025;      // RMS-scale inlet noise, F0 = A sqrt(M/2).
double fcut_dim = 28.0;     // Hz, cutoff frequency.
double * noise_phase = NULL;
double inlet_F = 0.;
double inlet_U = 0.;
int noise_seed = 12345;

// Non-dimensional variables
double fcut_noise, T0, H0, U0, US, RHO_RATIO, MU_RATIO; 

double chang_noise_signal (double tt)
{
  double F = 0.;
  double df = fcut_noise/((double) MNOISE);

  for (int k = 1; k <= MNOISE; k++) {
    double fk = k*df;
    F += cos(2.*pi*fk*tt + noise_phase[k-1]);
  }

  // Normalized so F0_noise is approximately the RMS amplitude.
  return F0_noise*sqrt(2./((double) MNOISE))*F;
}

u.n[left]  = dirichlet(inlet_U*(f0[]*profile[]));
//u.n[left]  = dirichlet(f0[]*profile[]);
u.t[left]  = dirichlet(0);
//p[left]   = neumann(0);
//pf[left]   = neumann(0);
//f[left]    = dirichlet(f0[]); 
d[left] = dirichlet(1.0-y);

//u.n[right] = neumann(0.);
//u.t[right] = neumann(0.);
u.t[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
u.n[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
//u.t[right] = neumann(0);
//p[right] = neumann(0);
//pf[right] = neumann(0); 
//f[right] = (u.x[] < 0.) ? dirichlet(0.):neumann(0); 

u.n[bottom] = dirichlet(0);
u.t[bottom] = dirichlet(0);
//f[bottom] = dirichlet(1);
//d[bottom] = dirichlet(1.0-y);

/*u.n[top] = dirichlet(0.);
u.t[top] = dirichlet(0.);
p[top] = dirichlet(0.);*/

void read_params(const char * fname);

int main (int argc, char * argv[])
{
  const char * fname = "params.in";
  if (argc > 1)
    fname = argv[1];

  TOLERANCE = 1e-4;
  NITERMIN = 2;
  NITERMAX = 100;
  CFL = 0.25;
  //  DT = 5e-5;

  read_params(fname);

  if (MNOISE <= 0) {
    fprintf(stderr, "MNOISE must be greater than zero\n");
    return 1;
  }

  noise_phase = malloc(MNOISE*sizeof(double));
  if (noise_phase == NULL) {
    fprintf(stderr,
	    "Could not allocate noise_phase for MNOISE = %d\n",
	    MNOISE);
    return 1;
  }

  H0 = cbrt(3.*RE*MU_L*MU_L/(RHO_L*RHO_L*grav*sin(angle)));
  U0 = grav*sin(angle)*H0*H0*RHO_L/MU_L/3.0;
  US = 1.5*U0;
  T0 = H0/U0;

  fcut_noise = fcut_dim*T0;

  size(LX);
  dimensions(nx = AR, ny = 1);
  
  init_grid(1<<MAXlevel);
  X0 = 0;
  Y0 = 0;

  
  rho1 = 1., rho2 = RHO_G/RHO_L;
  mu1 = 1./RE, mu2 = MU_G/MU_L/RE;

  const scalar sigma[] = 1./(RE*CA);
  d.sigmaf = sigma;
  /*f.sigma = GAMMA;
    f.height = h;*/

  //G.x = grav*sin(angle);
  //G.y = -grav*cos(angle);
  //Z.y = H0;
  G[0] = 3./RE;
  G[1] = -(3./RE)/tan(angle);
  
  char comm[80];
  sprintf(comm, "mkdir -p images");
  system(comm);
  
  sprintf(comm, "mkdir -p output");
  system(comm);

  sprintf(comm, "mkdir -p infc");
  system(comm);

  fprintf(stderr, "LX: %.8f\n", LX);
  fprintf(stderr, "LX [m]: %.8f\n", LX*H0);
  fprintf(stderr, "MAXlevel: %d\n", MAXlevel);
  fprintf(stderr, "U0 [m/s]: %.8f\n", U0);
  fprintf(stderr, "H0 [m/s]: %.8f\n", H0);
  fprintf(stderr, "T0 [s]: %.8f\n", T0);
  fprintf(stderr, "f_cut [Hz]: %.8f\n", fcut_dim);
  fprintf(stderr, "f : %.8f\n", fcut_noise);

  run();
}

void read_params(const char * fname)
{
  FILE * fp;
  if ((fp = fopen(fname, "rt"))) {
    char line[100];
    char key[80], val[80];

    while(fgets(line,100,fp)) {
      sscanf(line, "%15s = %15s", key, val);
      if      (strcmp(key,"LX") == 0)         { LX         =  atof(val);        }
      else if (strcmp(key, "MAXLEVEL") == 0)  { MAXlevel   = atoi(val);         }
      else if (strcmp(key, "AR") == 0)        { AR         = atoi(val);         }
      else if (strcmp(key, "Zoom") == 0)      { zoomy      = atoi(val);         }
      else if (strcmp(key, "CFL") == 0)       { CFL        = atof(val);         }
      else if (strcmp(key, "DT") == 0)        { DT         = atof(val);         }
      else if (strcmp(key, "TOLERANCE") == 0) { TOLERANCE  = atof(val);         }
      else if (strcmp(key, "CA") == 0)        { CA         = atof(val);         }
      else if (strcmp(key, "RE") == 0)        { RE         = atof(val);         }
      else if (strcmp(key, "ANGLE_DEG") == 0) { angle      = atof(val)*pi/180.; }
      //      else if (strcmp(key, "FREQ") == 0)      { freq_dim   = atof(val);         }
      //      else if (strcmp(key, "AMP") == 0)       { au         = atof(val);         }
      else if (strcmp(key, "T_OUT") == 0)     { t_out      = atof(val);         }
      else if (strcmp(key, "T_DUMP") == 0)    { t_dump     = atof(val);         }      
      else if (strcmp(key, "T_END") == 0)     { t_end      = atof(val);         }
      else if (strcmp(key, "F0_NOISE") == 0)  { F0_noise   = atof(val);         }
      else if (strcmp(key, "FCUT_DIM") == 0)  { fcut_dim   = atof(val);         }
      else if (strcmp(key, "NOISE_SEED") == 0){ noise_seed = atoi(val);         }
      else if (strcmp(key, "MNOISE") == 0)    { MNOISE     = atoi(val);         }
    }
    fclose(fp);
  } else {
    fprintf(stdout, "file %s not found\n", fname);
    //exit(0);
  }
}



event init (t = 0) {
  if (restore (file = "dump", list=all)) { 
  }
}

event init_noise (i = 0)
{
  srand(noise_seed);

  for (int k = 0; k < MNOISE; k++)
    noise_phase[k] = 2.*pi*(rand()/((double) RAND_MAX));

  inlet_F = chang_noise_signal(t);
  inlet_U = 1.5*(1. + inlet_F);
}

event update_inlet_noise (i++)
{
  inlet_F = chang_noise_signal(t);
  inlet_U = 1.5*(1. + inlet_F);
}

event inlet_signal_output (i++)
{
  if (pid() == 0) {
    static FILE * fp = NULL;

    if (fp == NULL) {
      fp = fopen("inlet_signal.dat", "w");
      fprintf(fp, "# t inlet_F inlet_U inlet_U_dim\n");
    }

    fprintf(fp, "%.12e %.12e %.12e %.12e\n",
            t, inlet_F, inlet_U, inlet_U*U0);
    fflush(fp);
  }
}

event check_grid(i=1)
{
  double xmax=0., ymax = 0., maxDelta = 0., minDelta = 10.;
  foreach(reduction(max:xmax) reduction(max:ymax) reduction(max:maxDelta) reduction(min:minDelta)){
    if(x > xmax) xmax = x;
    if(y > ymax) ymax = y;
    if(maxDelta < Delta) maxDelta = Delta;
    if(minDelta > Delta) minDelta = Delta;
  }

  fprintf(stderr, "N: %ld\n", grid->tn);
  fprintf(stderr, "Delta: %g , %g\n", maxDelta, minDelta);
  fprintf(stderr, "Domain: \nx : %g -> %g. \ny : %g -> %g\n", X0, xmax, Y0, ymax);
}

event acceleration (i++) {
  face vector av = a;
  foreach_face(x){
    av.x[] += G[0];
  }
  foreach_face(y){
    av.y[] += G[1];
  }
}

void mg_print (mgstats mg)
{
  if (mg.i > 0 && mg.resa > 0.)
    fprintf (stdout, " \t - \t %d %g %g %g %d ", mg.i, mg.resb, mg.resa,
	    mg.resb > 0 ? exp (log (mg.resb/mg.resa)/mg.i) : 0.,
	    mg.nrelax);
}


event logfile (i++) {
  if (i == 0)
    fprintf (stderr,
	     "t dt mgp.i mgpf.i mgu.i grid->tn perf.t perf.speed\n");
  fprintf (stderr, "%g %g %d %d %d %ld %g %g\n", 
	   t, dt, mgp.i, mgpf.i, mgu.i,
	   grid->tn, perf.t, perf.speed);
  fprintf (stdout, "\nPressure Residuals ");
  mg_print (mgp);
  fprintf (stdout, "\nVelocity Residuals ");
  mg_print (mgu);
  fprintf (stdout, "\n");
  fflush (stdout);
}


event interfacevel (t += t_out)
{
  char name[80];

  clear();
  view (tx = -0.5, ty = -0.5, sy = zoomy);
  draw_vof ("f", lw = 2);
  squares ("u.x", min = 0, max = 2, linear = true);
  colorbar(min = 0, max = 2);
  //isoline ("u.x", 1., lc = {1,1,1}, lw = 2);
  sprintf (name, "images/ux-%06.0f.png", t);
  save (name);
}

/*event interface (t += t_out) {

   char names[80];
   sprintf(names, "infc/interface%d", pid());
   FILE * fp = fopen (names, "w");
   output_facets (f,fp);
   fclose(fp);
   char command[80];
   sprintf(command, "LC_ALL=C  cat infc/interfa* > infc/infc%05.4f.dat",t);
   system(command);
}*/

/* event velocityprofile (t += 0.0001) {

   char names[80];
   sprintf(names, "velocity%d", pid());
   FILE * fp = fopen (names, "w");
   foreach() {
     fprintf(fp, "%.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
	     x, y, u.x[], u.y[], f[], d[], Delta);
   }

  fclose(fp);

   fclose(fp);
   char command[80];
   sprintf(command, "LC_ALL=C  cat velocity* > ux-vel%07.4f.dat",t);
   system(command);
} */

event finalize(t += t_dump; t <= t_end)
{
  char name[80];
  //scalar kappa[];
  //curvature(f, kappa);
  sprintf (name, "dump-%06.0f", t);
  p.nodump = false;
  dump (file = name); // so that we can restart
}

event output_h5(t += t_out; t<=t_end)
{
  char fname[256];
  scalar kappa[], kappa_minus[];

  sprintf(fname, "output/snapshot_%06.0f", t);

  output_xmf((scalar *){f,d,p,f_minus, d_minus, p_minus}, (vector *){u, u_minus}, fname);

  const char *parameter_names[] = {
    "Re",
    "Ca",
    "h0",
    "u0",
    "t0",
    "rho_ratio",
    "mu_ratio",
    "angle",
    "time",
    "time_minus",
    "dt"
  };

  double parameter_values[] = {
    RE,
    CA,
    H0,
    U0,
    T0,
    RHO_G/RHO_L,
    MU_G/MU_L,
    angle,
    t,
    t_minus,
    dt
  };

  int number_of_parameters =
    sizeof(parameter_values)/sizeof(parameter_values[0]);

  if (append_xmf_parameters (fname,
                             parameter_names,
                             parameter_values,
                             number_of_parameters) < 0) {
    if (pid() == 0)
      fprintf (stderr,
               "Could not append parameters to %s.h5\n",
               fname);
  }

  sprintf(fname, "infc/interface_%06.0f", t);
  foreach()
        kappa[] = distance_curvature (point, d);

  output_facets_xmf(f, kappa, fname);
  
  sprintf(fname, "infc/interface_minus_%06.0f", t);
  foreach()
        kappa_minus[] = distance_curvature (point, d_minus);

  output_facets_xmf(f_minus, kappa_minus, fname);
}


event save_previous (i++)
{
  t_minus = t;

  foreach() {
    f_minus[] = f[];
    p_minus[] = p[];
    d_minus[] = d[];
    foreach_dimension()
      u_minus.x[] = u.x[];
  }
} 

