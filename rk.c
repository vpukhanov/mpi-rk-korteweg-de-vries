#include <stdlib.h>
#include <mpi.h>

#include "rk.h"

void calc_step(double *, double *);

void runge_kutta(double *x, double dt, double t0, double t1, void report(double *, double)) {
    double *k1 = (double *) malloc(steps_num * sizeof(double));
    double *k2 = (double *) malloc(steps_num * sizeof(double));
    double *k3 = (double *) malloc(steps_num * sizeof(double));
    double *k4 = (double *) malloc(steps_num * sizeof(double));
    double *z = (double *) malloc(steps_num * sizeof(double));

    double t;

    for (t = t0; t <= t1; t += dt) {
        int i;

        /* Отправляем значения более старшим по рангу процессам */
        for (i = 0; i < mpi_rank; i++) {
            MPI_Send(&x[region_start], region_end - region_start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        /* Получаем значения от более младших по рангу процессов */
        for (i = mpi_rank + 1; i < mpi_size; i++) {
            MPI_Recv(&x[steps_num / mpi_size * i], region_end - region_start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, NULL);
        }

        /* Отправляем значения более младшим по рангу процессам */
        for (i = mpi_rank + 1; i < mpi_size; i++) {
            MPI_Send(&x[region_start], region_end - region_start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        /* Получаем значения от более старших по рангу процессов */
        for (i = 0; i < mpi_rank; i++) {
            MPI_Recv(&x[steps_num / mpi_size * i], region_end - region_start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, NULL);
        }

        if (report != NULL) {
            report(x, t);
        }

        calc_step(x, k1);

        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k1[i] / 2;
        }
        calc_step(z, k2);

        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k2[i] / 2;
        }
        calc_step(z, k3);

        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k3[i];
        }
        calc_step(z, k4);

        for (i = region_start; i < region_end; i++) {
            x[i] += (k1[i] + k2[i] * 2 + k3[i] * 2 + k4[i]) * dt / 6;
        }
    }

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(z);
}

void calc_step(double *x, double *dst) {
    int n = steps_num;
    int points = n / 50;
    int tpoints = points * points * points;

    int i = 0;
    dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1 + n]) * 0.5 * points
             - (x[i + 2] - 2 * x[i + 1] + 2 * x[i - 1 + n] - x[i - 2 + n]) * 0.5 * tpoints;

    i = 1;
    dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1]) * 0.5 * points
             - (x[i + 2] - 2 * x[i + 1] + 2 * x[i - 1] - x[i - 2 + n]) * 0.5 * tpoints;

    for (i = 2; i < n - 2; i++) {
        dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1]) * 0.5 * points
                 - (x[i + 2] - 2 * x[i + 1] + 2 * x[i - 1] - x[i - 2]) * 0.5 * tpoints;
    }

    i = n - 2;
    dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1]) * 0.5 * points
             - (x[i + 2 - n] - 2 * x[i + 1] + 2 * x[i - 1] - x[i - 2]) * 0.5 * tpoints;

    i = n - 1;
    dst[i] = -6 * x[i] * (x[i + 1 - n] - x[i - 1]) * 0.5 * points
             - (x[i + 2 - n] - 2 * x[i + 1 - n] + 2 * x[i - 1] - x[i - 2]) * 0.5 * tpoints;
}