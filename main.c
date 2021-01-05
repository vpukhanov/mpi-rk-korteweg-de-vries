#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "rk.h"

FILE *foutput;
int mpi_size, mpi_rank, region_start, region_end, steps_num;

void print_report(double *, double);

/* mpiexec -n <number of processes> main [points factor=10] */
int main(int argc, char **argv) {
    int points_factor = argc >= 2 ? atoi(argv[1]) : 10;

    steps_num = 50 * points_factor;

    double start = 0;
    double end = 10;
    double step = 0.001;

    double *x = (double *) malloc(steps_num * sizeof(double));

    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    region_start = steps_num / mpi_size * mpi_rank;
    region_end = steps_num / mpi_size * (mpi_rank + 1);

    for (i = region_start; i < region_end; i++) {
        double ox = i * 0.1;
        double oxcosh = cosh(ox - 25);
        x[i] = 2 / (oxcosh * oxcosh);
    }

    if (mpi_rank == 0) {
        foutput = fopen("output.txt", "w");
    }

    runge_kutta(x, step, start, end, mpi_rank == 0 ? print_report : NULL);

    if (mpi_rank == 0) {
        fclose(foutput);
    }

    free(x);
    MPI_Finalize();

    return 0;
}

void print_report(double *arr, double t) {
    int i;

    fprintf(foutput, "%.5f\n", t);
    for (i = 0; i < steps_num; i++) {
        fprintf(foutput, "%.5f ", arr[i]);
    }

    fprintf(foutput, "\n");
}