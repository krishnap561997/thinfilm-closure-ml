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
#include "output_vtu_foreach.h"
//vector h[];

double t_out = 0.01;
double t_dump = 0.1;
double t_end = 2.8;
double H0 = 0.001279;
double RE, CA, US, T0;
double GAMMA = 0.067;
double RHO_L = 1072,
  RHO_G = 1.225,
  MU_L = 0.00673,
  MU_G = 1.7894e-05;
double grav = 9.81;

double G[2];
double LX = 0.2;
int AR = 32, zoomy = 32;

int MAXlevel = 12;
//double uemax = 0.0001;
#define NXCOL (1 << MAXlevel)

double angle = 6.4*pi/180.;
double au = 0.03, freq = 1.5;
scalar f0[], profile[];

u.n[left]  = dirichlet(US*(1 + au*sin(2*pi*freq*t))*(f0[]*profile[])); // + (1-f0[]))));
//u.n[left]  = dirichlet(f0[]*profile[]);
u.t[left]  = dirichlet(0);
//p[left]   = neumann(0);
//pf[left]   = neumann(0);
//f[left]    = dirichlet(f0[]); 
d[left] = dirichlet(H0-y);

u.n[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
u.t[right] = f[] > 1e-6 ? neumann(0.):dirichlet(0.) ;
//u.t[right] = neumann(0);
p[right] = dirichlet(0);
pf[right] = dirichlet(0); 
//f[right] = neumann(0);

u.n[bottom] = dirichlet(0);
u.t[bottom] = dirichlet(0);
//f[bottom] = dirichlet(1);
d[bottom] = dirichlet(H0-y);

/*u.n[top] = dirichlet(0.);
u.t[top] = dirichlet(0.);
p[top] = dirichlet(0.);*/

void read_params(const char * fname);
void compute_derivative( double a[], double ax[], double axx[], double axxx[], int N, double dx);

int main (int argc, char * argv[])
{
  const char * fname = "params.in";
  if (argc > 1)
    fname = argv[1];

  TOLERANCE = 1e-4;
  NITERMIN = 2;
  NITERMAX = 100;
  CFL = 0.25;
  DT = 5e-5;

  read_params(fname);
  
  US = grav*sin(angle)*H0*H0*RHO_L/MU_L/2.0;
  RE = US*H0*RHO_L/MU_L;
  CA = MU_L*US/GAMMA;
  T0 = 1/freq;  

  size(LX);
  dimensions(nx = AR, ny = 1);
  
  init_grid(1<<MAXlevel);
  X0 = 0;
  Y0 = 0;

  
  rho1 = RHO_L, rho2 = RHO_G;
  mu1 = MU_L, mu2 = MU_G;

  const scalar sigma[] = GAMMA;
  d.sigmaf = sigma;
  /*f.sigma = GAMMA;
    f.height = h;*/

  //G.x = grav*sin(angle);
  //G.y = -grav*cos(angle);
  //Z.y = H0;
  G[0] = grav*sin(angle);
  G[1] = -grav*cos(angle);

  char comm[80];
  sprintf(comm, "mkdir -p images");
  system(comm);


  fprintf(stderr, "LX: %.8f\n", LX);
  fprintf(stderr, "MAXlevel: %d\n", MAXlevel);
  fprintf(stderr, "Us: %.8f\n", US);
  fprintf(stderr, "Re: %.8f\n", RE);
  fprintf(stderr, "Ca: %.8f\n", CA);
  fprintf(stderr, "T0: %.8f\n", T0);

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
      else if (strcmp(key, "T_OUT") == 0)     { t_out     = atof(val);         }
      else if (strcmp(key, "T_END") == 0)     { t_end     = atof(val);         }
    }
    fclose(fp);
  } else {
    fprintf(stdout, "file %s not found\n", fname);
    exit(0);
  }
}

void compute_derivatives (
  double a[], double ax[], double axx[], double axxx[],
  int N, double dx
) {
  // Interior points
  for (int i = 2; i < N - 2; i++) {
    ax[i] =
      (a[i+1] - a[i-1])/(2.0*dx);

    axx[i] =
      (a[i+1] - 2.0*a[i] + a[i-1])/(dx*dx);

    axxx[i] =
      (a[i+2] - 2.0*a[i+1] + 2.0*a[i-1] - a[i-2])
      /(2.0*dx*dx*dx);
  }

  // Boundary treatment: Copying nearest value
  ax[0] = ax[1] = ax[2];
  ax[N-2] = ax[N-1] = ax[N-3];

  axx[0] = axx[1] = axx[2];
  axx[N-2] = axx[N-1] = axx[N-3];

  axxx[0] = axxx[1] = axxx[2];
  axxx[N-2] = axxx[N-1] = axxx[N-3];
}



event init (t = 0) {
  if (!restore (file = "dump")) { 
    fraction (f0, H0 - y);
    //f0.refine = f0.prolongation = fraction_refine;
    restriction ({f0}); // for boundary conditions on levels

    foreach(){
      profile[] = (y/H0)*(2.0-(y/H0));
    }
    //profile.refine = profile.prolongation = refine_linear;
    //profile.refine = profile.prolongation = fraction_refine;
    restriction ({profile}); // for boundary conditions on levels
   
   
    foreach() {
      //f[] = f0[];
      d[] = H0 - y;
      u.x[] = US*(f0[]*profile[]); // + 1-f0[]);
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

/*event damp (i++) {
  coord Uinf = {US, 0, 0};
  foreach() {
    if (LX - x < LX/10.)
      foreach_dimension()
        u.x[] += dt*(Uinf.x*profile[]*f0[] - u.x[])/2.;
  }
  boundary ((scalar*){u});
}*/


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
  squares ("u.x", min = 0, max = 1.5*US, linear = true);
  colorbar(min = 0, max = 1.5*US);
  //isoline ("u.x", 1., lc = {1,1,1}, lw = 2);
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
   sprintf(names, "interface%d", pid());
   FILE * fp = fopen (names, "w");
   output_facets (f,fp);
   fclose(fp);
   char command[80];
   sprintf(command, "LC_ALL=C  cat interfa* > infc%05.4f.dat",t);
   system(command);
}

event closure_quantities (t+=0.001) {

  double h[NXCOL], q[NXCOL];

  for (int i = 0; i < NXCOL; i++) {
    h[i] = 0.;
    q[i] = 0.;
  }

  double xmin = 0.;
  double xmax = L0;
  double dxcol = (xmax - xmin)/NXCOL;

  foreach(reduction(+:h[:NXCOL]) reduction(+:q[:NXCOL])) {

    double xleft  = x - Delta/2.;
    double xright = x + Delta/2.;

    int i0 = floor((xleft  - xmin)/dxcol);
    int i1 = floor((xright - xmin)/dxcol);

    if (i0 < 0) i0 = 0;
    if (i1 >= NXCOL) i1 = NXCOL - 1;

    for (int i = i0; i <= i1; i++) {
      h[i] += f[]*Delta;
      q[i] += f[]*u.x[]*Delta;
    }
  }

  double hx[NXCOL], hxx[NXCOL], hxxx[NXCOL];
  double qx[NXCOL], qxx[NXCOL], qxxx[NXCOL];

  compute_derivatives(h, hx, hxx, hxxx, NXCOL, dxcol);
  compute_derivatives(q, qx, qxx, qxxx, NXCOL, dxcol);

  if (pid() == 0) {
    char name[80];
    sprintf(name, "hq-%g.dat", t);

    FILE * fp = fopen(name, "w");

    fprintf(fp,
      "# t x h q hx hxx hxxx qx qxx qxxx\n"
    );

    for (int i = 0; i < NXCOL; i++) {
      double xc = xmin + (i + 0.5)*dxcol;

      fprintf(fp,
        "%.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
        t, xc,
        h[i], q[i],
        hx[i], hxx[i], hxxx[i],
        qx[i], qxx[i], qxxx[i]
      );
    }

    fclose(fp);
  } 
} 


/*event velocityprofile (t += 0.0001) {

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


/*event velocityprofile (t += 0.0001) {

  char name[80];
  sprintf(name, "vel-%07.4f.vtu", t);

  FILE * fp = fopen(name, "w");

  scalar omega[];
  vorticity(u, omega);

  output_vtu_ascii_foreach(
    (scalar *) {f, d, p, omega},
    (vector *) {u},
    fp
  );

  fclose(fp);
  }*/


/* event movie (t += 1e-2)
{
#if dimension == 2
  scalar omega[];
  vorticity (u, omega);
  view (tx = -0.5);
  clear();
  draw_vof ("f");
  squares ("omega", linear = true, spread = 10);
  box ();
#else // 3D
  scalar pid[];
  foreach()
    pid[] = fmod(pid()*(npe() + 37), npe());
  view (camera = "iso",
	fov = 14.5, tx = -0.418, ty = 0.288,
	width = 1600, height = 1200);
  clear();
  draw_vof ("f");
#endif // 3D
  save ("movie.mp4");
  } */

/*event snapshot (t = 0; t += 10; t <= 300) {
  char name[80];
  sprintf (name, "snapshot-%g", t);
  scalar pid[];
  foreach()
    pid[] = fmod(pid()*(npe() + 37), npe());
  dump (name);
  } */

/*event snapshot (t += t_out; t<=t_end) {
  char name[80];
  scalar kappa[];
  curvature(f, kappa);
  sprintf (name, "dump-%06.4f", t);
  p.nodump = false;
  dump (file = name); // so that we can restart
}*/

event finalize(t += t_dump; t <= t_end)
{
  char name[80];
  //scalar kappa[];
  //curvature(f, kappa);
  sprintf (name, "dump-%06.4f", t);
  p.nodump = false;
  dump (file = name); // so that we can restart
}

/*event runtime (i += 10) {
  mpi_all_reduce (perf.t, MPI_DOUBLE, MPI_MAX);
  if (perf.t/60 >= maxruntime) {
    dump (file = "dump"); // so that we can restart
    return 1; // exit
  }
  }*/

