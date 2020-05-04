# gdtomo
gdtomo is a gradient-descent based tomographic algorithm written in C with cuda to minimize memory footprint and optimize with GPU parallelization.

## Installation
In the main directory, call
```bash
make
```
to build gdtomo in C. The current version is tested with
  - gcc 7.4.0
  - x86_64 Linux 5.6.8-arch1-1
  - nvcc 10.2

Invoke
```bash
make clean
```
to clean the current build of gdtomo.

## Usage
Call
```bash
./gdtomo recon recon_info.txt
```
to perform tomography. A sample of ```recon_info.txt``` is shown below.

```
[projections_filename]
./example/projs.npy

[angles_filename]
./example/angles.npy

[recon_filename]
./example/recon.npy

[err_filename]
./example/err.npy

[n_iter]
200

[alpha]
1.0

[recon_dim_x]
50

[recon_dim_y]
50

[recon_dim_z]
50

```

## Parameters

Let *N* be the number of projections, *D1* and *D2* be the projection dimensions.

``projections_filename`` points to a 3-dimensional projections dataset with dimensions [*N*,*D1*,*D2*]. The projections are centered so that the rotational center is located at [(*D1*+1)/2,(*D2*+1)/2]. The file should be a 3D matrix in ```.npy``` format.

``angles_filename`` points to a 2-dimensional angle dataset with dimensions [*N*,3]. These are euler angles (Z-Y-X convention, with Z being the first rotation) of the projection rotations in degrees. The file should be a 2D matrix in ```.npy``` format.

``recon_filename`` is where the final reconstruction will be saved in ```.npy``` format.

``err_filename`` is where the L1 and L2 norms of the errors will be saved in ```.npy``` format.

``n_iter`` is the number of iterations.

``alpha`` is the rate of gradient descent. Increasing this value will increase speed of reconstruction.

``recon_dim_*`` are the dimensions of the final reconstruction.
