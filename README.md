# gdtomo
gdtomo is a gradient-descent based tomographic algorithm written in C to
minimize memory footprint and optimized with parallelization.

## Installation
In the main directory, invoke
```bash
make
```
to build gdtomo in C. The current version is tested with gcc 7.4.0 in Ubuntu
7.4.0-1ubuntu1~18.04.1.

Invoke
```bash
make clean
```
to clean the current build of gdtomo.

## Usage
A wrapper is written in MATLAB found in the ``matlab`` directory. To use
gdtomo, make sure to update Line 1 of ``gdtomo.m`` or ``calc_projs.m`` with
the correct path to gdtomo.
```matlab
GDTOMO_PATH = '../'; % <-- modify this
```
By doing so, you'll be able to call gdtomo even though the matlab files exist
in another directory.

## Parameters
The parameters of the reconstruction can be modified in the first few lines of gdtomo.m:
```matlab
% === PARAMETERS ===============================
projs_filename  = "./data/test_gdtomo/projs.mat" ;
angles_filename = "./data/test_gdtomo/angles.mat";
recon_filename  = "./data/test_gdtomo/recon.mat" ;
err_filename    = "./data/test_gdtomo/err.mat"   ;
num_iter        = 50;
recon_alpha     = 0.5;
recon_dim       = [50,50,50];
num_cores       = 6;
% ==============================================
```

Let *N* be the number of projections, *D1* and *D2* be the projection dimensions.

``projs_filename`` points to a 3-dimensional dataset with dimensions
[*N*,*D1*,*D2*]. The projections are centered so that the rotational center
is located at [(*D1*+1)/2,(*D2*+1)/2].

``angles_filename`` points to a 2-dimensional dataset with dimensions [*N*,3]. These are euler angles of the projection rotations in degrees.

``recon_filename`` is where the final reconstruction will be saved.

``err_filename`` is where the L1 and L2 norms of the errors are saved.

``recon_alpha`` is the rate of gradient descent in the algorithm. Increasing this value will increase speed of reconstruction.

``recon_dim`` are the dimensions of the final reconstruction.

``num_cores`` sets the number of CPU cores you want to utilize during reconstruction.
