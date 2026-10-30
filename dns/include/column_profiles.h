/*
 * column_profiles.h -- in-situ depth-integrated (1D) output for thin-film runs.
 *
 * Replaces the 2D snapshot when only the depth-integrated quantities are needed.
 * For every x-column of the uniform multigrid it computes, for (f,u,p) and for the
 * previous-step fields (f_minus,u_minus,p_minus):
 *
 *   h               = sum_j f dy
 *   q               = sum_j f u dy
 *   int_u2_dy       = sum_j f u^2 dy
 *   int_px_dy       = int_0^h dp/dx dy   (4th-order x-stencil, trapezoid to y = h)
 *   dudy_wall       = (3675 u1 - 1225 u2 + 441 u3 - 75 u4)/(840 dy)
 *   u_interface_raw = u linearly interpolated to y = h
 *
 * The definitions follow pipeline/extract_closures/extract_closure_terms.py
 * (depth_integral, integrate_to_height, compute_px, compute_wall_gradient and the
 * raw step of compute_ui), so derivatives and closures are computed from these 1D
 * profiles by the same Python code. Uniform multigrid with X0 = Y0 = 0, OpenMP only.
 */
#ifndef COLUMN_PROFILES_H
#define COLUMN_PROFILES_H

#include <hdf5.h>
#include <hdf5_hl.h>

#define CP_NFIELD 6
static const char * cp_names[CP_NFIELD] = {
  "h", "q", "int_u2_dy", "int_px_dy", "dudy_wall", "u_interface_raw"
};

static int cp_nx = 0, cp_ny = 0;
static double cp_dl = 0.;
static double * cp_F = NULL, * cp_U = NULL, * cp_P = NULL, * cp_PX = NULL;

static void column_profiles_init ()
{
  double xmax = 0., ymax = 0., dmax = 0.;
  foreach (reduction(max:xmax) reduction(max:ymax) reduction(max:dmax)) {
    if (x > xmax) xmax = x;
    if (y > ymax) ymax = y;
    if (Delta > dmax) dmax = Delta;
  }
  cp_dl = dmax;
  cp_nx = (int) ((xmax - X0)/cp_dl) + 1;   /* last centre at (n - 1/2) dl */
  cp_ny = (int) ((ymax - Y0)/cp_dl) + 1;
  if ((long) cp_nx*cp_ny != grid->tn) {
    fprintf (stderr, "column_profiles: %d x %d cells but grid->tn = %ld"
             " (needs a uniform grid)\n", cp_nx, cp_ny, grid->tn);
    exit (1);
  }
  size_t n = (size_t) cp_nx*cp_ny;
  cp_F = malloc (n*sizeof(double)); cp_U = malloc (n*sizeof(double));
  cp_P = malloc (n*sizeof(double)); cp_PX = malloc (n*sizeof(double));
  if (!cp_F || !cp_U || !cp_P || !cp_PX) {
    fprintf (stderr, "column_profiles: allocation failed\n");
    exit (1);
  }
  fprintf (stderr, "column_profiles: %d x %d cells, dy = %g\n", cp_nx, cp_ny, cp_dl);
}

/* np.interp(xq, xs, fs) on cell centres ys_j = (j + 1/2) dy, clamped at the ends. */
static double cp_interp_column (const double * col, double yq)
{
  double s = yq/cp_dl - 0.5;
  if (s <= 0.) return col[0];
  if (s >= cp_ny - 1) return col[(size_t) (cp_ny - 1)*cp_nx];
  int j = (int) s;
  double w = s - j;
  return (1. - w)*col[(size_t) j*cp_nx] + w*col[(size_t) (j + 1)*cp_nx];
}

/* out[k] has cp_nx entries, ordered as cp_names. */
static void column_profiles_compute (scalar ff, vector uu, scalar pp, double * out[CP_NFIELD])
{
  if (!cp_F) column_profiles_init ();
  const int nx = cp_nx, ny = cp_ny;
  const double dl = cp_dl;
  double * F = cp_F, * U = cp_U, * P = cp_P, * PX = cp_PX;

  foreach () {
    int i = (int) ((x - X0)/Delta), j = (int) ((y - Y0)/Delta);
    size_t k = (size_t) j*nx + i;
    F[k] = ff[]; U[k] = uu.x[]; P[k] = pp[];
  }

  /* dp/dx along each row: compute_px / first_derivative_x */
#pragma omp parallel for
  for (int j = 0; j < ny; j++) {
    const double * r = P + (size_t) j*nx;
    double * o = PX + (size_t) j*nx;
    for (int i = 2; i < nx - 2; i++)
      o[i] = (r[i-2] - 8.*r[i-1] + 8.*r[i+1] - r[i+2])/(12.*dl);
    o[0] = (-25.*r[0] + 48.*r[1] - 36.*r[2] + 16.*r[3] - 3.*r[4])/(12.*dl);
    o[1] = (-3.*r[0] - 10.*r[1] + 18.*r[2] - 6.*r[3] + r[4])/(12.*dl);
    o[nx-2] = (-r[nx-5] + 6.*r[nx-4] - 18.*r[nx-3] + 10.*r[nx-2] + 3.*r[nx-1])/(12.*dl);
    o[nx-1] = (3.*r[nx-5] - 16.*r[nx-4] + 36.*r[nx-3] - 48.*r[nx-2] + 25.*r[nx-1])/(12.*dl);
  }

#pragma omp parallel for
  for (int i = 0; i < nx; i++) {
    double h = 0., q = 0., m = 0.;
    for (int j = 0; j < ny; j++) {
      size_t k = (size_t) j*nx + i;
      h += F[k]*dl; q += F[k]*U[k]*dl; m += F[k]*U[k]*U[k]*dl;
    }
    /* integrate_to_height: trapezoid over [0, centres with 0 < y < h, h] */
    const double * pc = PX + i;
    double yp = 0., fp = pc[0], ip = 0.;
    for (int j = 0; j < ny; j++) {
      double yj = (j + 0.5)*dl;
      if (yj >= h) break;
      double fj = pc[(size_t) j*nx];
      ip += 0.5*(fp + fj)*(yj - yp);
      yp = yj; fp = fj;
    }
    ip += 0.5*(fp + cp_interp_column (pc, h))*(h - yp);

    const double * uc = U + i;
    out[0][i] = h; out[1][i] = q; out[2][i] = m; out[3][i] = ip;
    out[4][i] = (3675.*uc[0] - 1225.*uc[nx] + 441.*uc[2*nx] - 75.*uc[3*nx])/(840.*dl);
    out[5][i] = cp_interp_column (uc, h);
  }
}

static int column_profiles_write_param (hid_t g, const char * name, double v)
{
  hsize_t one = 1;
  return H5LTmake_dataset_double (g, name, 1, &one, &v);
}

/*
 * Write one profile file: x, the six fields at t and (suffix _m) at t_minus, plus
 * parameters/{...}. names/values are the run parameters (Re, Ca, angle, ...).
 */
static int output_column_profiles (const char * fname,
                                   scalar ff, vector uu, scalar pp,
                                   scalar fm, vector um, scalar pm,
                                   const char ** names, const double * values, int npar)
{
  if (!cp_F) column_profiles_init ();
  double * buf = malloc ((size_t) (2*CP_NFIELD + 1)*cp_nx*sizeof(double));
  if (!buf) { fprintf (stderr, "column_profiles: allocation failed\n"); return -1; }
  double * cur[CP_NFIELD], * prv[CP_NFIELD], * xs = buf + (size_t) 2*CP_NFIELD*cp_nx;
  for (int k = 0; k < CP_NFIELD; k++) {
    cur[k] = buf + (size_t) k*cp_nx;
    prv[k] = buf + (size_t) (CP_NFIELD + k)*cp_nx;
  }
  for (int i = 0; i < cp_nx; i++) xs[i] = X0 + (i + 0.5)*cp_dl;

  column_profiles_compute (ff, uu, pp, cur);
  column_profiles_compute (fm, um, pm, prv);

  int status = 0;
  hid_t file = H5Fcreate (fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) { free (buf); return -1; }
  hsize_t n = cp_nx;
  status |= H5LTmake_dataset_double (file, "x", 1, &n, xs);
  char name[64];
  for (int k = 0; k < CP_NFIELD; k++) {
    status |= H5LTmake_dataset_double (file, cp_names[k], 1, &n, cur[k]);
    snprintf (name, sizeof(name), "%s_m", cp_names[k]);
    status |= H5LTmake_dataset_double (file, name, 1, &n, prv[k]);
  }
  hid_t g = H5Gcreate2 (file, "parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  for (int k = 0; k < npar; k++)
    status |= column_profiles_write_param (g, names[k], values[k]);
  H5Gclose (g);
  H5Fclose (file);
  free (buf);
  return status < 0 ? -1 : 0;
}

#endif // COLUMN_PROFILES_H
