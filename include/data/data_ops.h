#ifndef DATA_OPS_H
#define DATA_OPS_H

#include <data/data_types.h>
#define PI 3.14159265

/*
Calculates the euler rotation of [v] with euler [angles], in theta, phi, psi for
Z-Y-X rotation, and saves it as [v_f]
*/
void euler_rot(float* v, float* angles, float* v_f);

/*
Calculates the reverse euler rotation of [v] with euler [angles], in theta, phi,
psi for Z-Y-X rotation, and saves it as [v_f]
*/
void euler_rot_rev(float* v, float* angles, float* v_f);

/* Converts from degrees to radians */
void deg_to_rad(Data_2d* angles);
void rad_to_deg(Data_2d* angles);

int in_bounds(int i, int Ni);

#endif
