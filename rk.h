#ifndef RK_H
#define RK_H

extern int mpi_size, mpi_rank, region_start, region_end, steps_num;

void runge_kutta(double *, double, double, double, void (double *, double));

#endif