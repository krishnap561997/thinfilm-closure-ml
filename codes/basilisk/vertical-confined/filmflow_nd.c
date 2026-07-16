#include "grid/multigrid.h"
#include "axi.h"
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

//#include "hdf5_headers/output_xdmf.h"

double t_out_dim = 1;
double t_dump_dim = 0.1;
double t_end_dim = 2.8;

double t_out, t_dump, t_end;

double H0 = 0.001279;
double RE, CA, US, T0, U0;
double GAMMA = 0.067;
double RHO_L = 1072,
  RHO_G = 1.225,
  MU_L = 0.00673,
  MU_G = 1.7894e-05;
double grav = 9.81;

double G[2];
double LX = 128;
double LY;
int AR = 32, zoomy = 32;

int MAXlevel = 12;
//double uemax = 0.0001;

double angle = 90*pi/180.;
double au = 0.03, freq = 1.5;
double t_start = 1.0, t_slope = 10; //Ramp to prevent initial high amp wave
scalar f0[], profile[];

double sigmoid(double t1, double k);

u.n[left]  = dirichlet((1 + au*sigmoid(t_start/T0, t_slope*T0)*sin(2*pi*freq*t*T0))*(f0[]*profile[])); 
//u.n[left]  = dirichlet(f0[]*profile[]);
u.t[left]  = dirichlet(0);
//p[left]   = neumann(0);
//pf[left]   = neumann(0);
//f[left]    = dirichlet(f0[]); 
d[left] = dirichlet(y + 1.0 - LY);

u.n[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
u.t[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
//u.t[right] = neumann(0);
p[right] = dirichlet(0);
pf[right] = dirichlet(0); 
//f[right] = neumann(0);

u.n[top] = dirichlet(0);
u.t[top] = dirichlet(0);
//f[bottom] = dirichlet(1);
d[top] = dirichlet(1.0);

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
  
  US = grav*sin(angle)*H0*H0*RHO_L/MU_L/2.0;
  U0 = 2.0*US/3.0;
  RE = U0*H0*RHO_L/MU_L;
  CA = MU_L*U0/GAMMA;
  T0 = H0/U0;
  LY = LX/((double)AR);

  /*t_out = t_out_dim/T0;
  t_end = t_end_dim/T0;
  t_dump = t_dump_dim/T0;*/

  t_out = 1.0;
  t_end = 10.0;
  t_dump = 1.0;
  
  size(LX);
  dimensions(nx = AR, ny = 1);
  
  init_grid(1<<MAXlevel);
  X0 = 0;
  Y0 = 0;

  
  rho1 = RE;
  rho2 = RHO_G*rho1/RHO_L;
  mu1 = 1.0;
  mu2 = MU_G/MU_L;

  const scalar sigma[] = 1.0/CA;
  d.sigmaf = sigma;
  /*f.sigma = GAMMA;
    f.height = h;*/

  //G.x = grav*sin(angle);
  //G.y = -grav*cos(angle);
  //Z.y = H0;
  G[0] = 3.0/RE;
  G[1] = 0.;
  
  char comm[80];
  sprintf(comm, "mkdir -p images");
  system(comm);
  
  sprintf(comm, "mkdir -p output");
  system(comm);

  sprintf(comm, "mkdir -p infc");
  system(comm);

  fprintf(stderr, "LX: %.8f\n", LX);
  fprintf(stderr, "MAXlevel: %d\n", MAXlevel);
  fprintf(stderr, "Us: %.8f\n", US);
  fprintf(stderr, "Re: %.8f\n", RE);
  fprintf(stderr, "Ca: %.8f\n", CA);
  fprintf(stderr, "T0: %.8f\n", T0);
  fprintf(stderr, "TDUMP: %.8f\n", t_dump);
  fprintf(stderr, "TEND: %.8f\n", t_end);

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
      if (strcmp(key,"LX") == 0)              { LX        = atof(val);         }
      else if (strcmp(key, "MAXLEVEL") == 0)  { MAXlevel  = atoi(val);         }
      else if (strcmp(key, "AR") == 0)        { AR        = atoi(val);         }
      else if (strcmp(key, "Zoom") == 0)      { zoomy     = atoi(val);         }
      else if (strcmp(key, "CFL") == 0)       { CFL       = atof(val);         }
      else if (strcmp(key, "DT") == 0)        { DT        = atof(val);         }
      else if (strcmp(key, "TOLERANCE") == 0) { TOLERANCE = atof(val);         }
      else if (strcmp(key, "H0") == 0)        { H0        = atof(val);         }
      else if (strcmp(key, "ANGLE_DEG") == 0) { angle     = atof(val)*pi/180.; }
      else if (strcmp(key, "FREQ") == 0)      { freq      = atof(val);         }
      else if (strcmp(key, "AMP") == 0)       { au        = atof(val);         }
      else if (strcmp(key, "T_OUT") == 0)     { t_out_dim = atof(val);         }
      else if (strcmp(key, "T_DUMP") == 0)    { t_dump_dim= atof(val);         }
      else if (strcmp(key, "T_END") == 0)     { t_end_dim = atof(val);         }
      else if (strcmp(key, "SIGMOID_T1") == 0){ t_start   = atof(val);         }
      else if (strcmp(key, "SIGMOID_K") == 0) { t_slope   = atof(val);         }
    }
    fclose(fp);
  } else {
    fprintf(stdout, "file %s not found\n", fname);
    //exit(0);
  }
}

double sigmoid(double t1, double k) {
  return 1.0 / (1.0 + exp(-k * (t -t1)));
}

event init (t = 0) {
  if (!restore (file = "dump")) { 
    fraction (f0, y + 1.0 - LY);
    //f0.refine = f0.prolongation = fraction_refine;
    restriction ({f0}); // for boundary conditions on levels

    foreach(){
      profile[] = 1.5*(LY-y)*(2.0-(LY-y));
    }
    //profile.refine = profile.prolongation = refine_linear;
    //profile.refine = profile.prolongation = fraction_refine;
    restriction ({profile}); // for boundary conditions on levels
   
   
    foreach() {
      //f[] = f0[];
      d[] = y + 1.0 - LY;
      u.x[] = f0[]*profile[]; // + 1-f0[]);
      u.y[] = 0;
    }
    boundary({d, u});
    
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

  if (i==0)
  {
	clear();
        view (tx = -0.5, ty = -0.5, sx = zoomy, sy = 2*zoomy);
	draw_vof ("f", lw = 6);
	cells ();
	sprintf (name, "images/dimcheck-%5.4f.png", t);
	save (name);      
  }
  clear();
  view (tx = -0.5, ty = -0.5, sy = zoomy);
  draw_vof ("f", lw = 2);
  squares ("u.x", min = 0, max = 3, linear = true);
  colorbar(min = 0, max = 3);
  sprintf (name, "images/ux-%5.4f.png", t);
  save (name);

  clear();
  view (tx = -0.5, ty = -0.5, sy = 2*zoomy);
  draw_vof ("f", lw = 2);
  squares ("p", linear = true, spread=10);
  sprintf (name, "images/pfp-%5.4f.png", t);
  save (name);
}

event interface (t += t_out) {

   char names[80];
   sprintf(names, "infc/interface%d", pid());
   FILE * fp = fopen (names, "w");
   output_facets (f,fp);
   fclose(fp);
   char command[80];
   sprintf(command, "LC_ALL=C  cat infc/interfa* > infc/infc%05.4f.dat",t);
   system(command);
}

event finalize(t = 0; t += t_dump; t <= t_end)
{
  char name[80];
  sprintf (name, "dump-%06.4f", t);
  p.nodump = false;
  dump (file = name); // so that we can restart
}

/*event output_h5(t = 0; t += t_out; t <= t_end)
{
  char fname[256];
  sprintf(fname, "output/snapshot_%06.4f", t);

  output_xmf((scalar *){f,p}, (vector *){u}, fname);
}*/

event end(t = t_end)
