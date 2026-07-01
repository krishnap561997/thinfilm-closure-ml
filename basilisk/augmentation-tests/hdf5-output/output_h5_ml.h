#ifndef OUTPUT_H5_ML_H
#define OUTPUT_H5_ML_H

#include <hdf5.h>

static void write_1d_h5(hid_t file, const char *name, double *data, int n)
{
  hsize_t dims[1] = {n};

  hid_t space = H5Screate_simple(1, dims, NULL);
  hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
           H5P_DEFAULT, data);

  H5Dclose(dset);
  H5Sclose(space);
}

static void write_2d_h5(hid_t file, const char *name,
                        double *data, int ny, int nx)
{
  hsize_t dims[2] = {ny, nx};

  hid_t space = H5Screate_simple(2, dims, NULL);
  hid_t dset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
           H5P_DEFAULT, data);

  H5Dclose(dset);
  H5Sclose(space);
}

#endif
