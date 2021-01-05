#include <stdlib.h>

#include "rk.h"

void calc_step(double *, double *, int);

void runge_kutta(double *x, double dt, int steps_num, double t0, double t1, void report(double *, double, int)) {
    double *k1 = (double *) malloc(steps_num * sizeof(double));
    double *k2 = (double *) malloc(steps_num * sizeof(double));
    double *k3 = (double *) malloc(steps_num * sizeof(double));
    double *k4 = (double *) malloc(steps_num * sizeof(double));
    double *z = (double *) malloc(steps_num * sizeof(double));

    double t;

    for (t = t0; t <= t1; t += dt) {
        int i;

        report(x, t, steps_num);

        calc_step(x, k1, steps_num);

#pragma omp parallel for shared(z) private(i)
        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k1[i] / 2;
        }
        calc_step(z, k2, steps_num);

#pragma omp parallel for shared(z) private(i)
        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k2[i] / 2;
        }
        calc_step(z, k3, steps_num);

#pragma omp parallel for shared(z) private(i)
        for (i = 0; i < steps_num; i++) {
            z[i] = x[i] + dt * k3[i];
        }
        calc_step(z, k4, steps_num);

#pragma omp parallel for shared(x) private(i)
        for (i = 0; i < steps_num; i++) {
            x[i] += (k1[i] + k2[i] * 2 + k3[i] * 2 + k4[i]) * dt / 6;
        }
    }

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(z);
}

void calc_step(double *x, double *dst, int n) {
    int points = n / 50;
    int tpoints = points * points * points;

    int i = 0;
    dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1 + n]) * 0.5 * points
             - (x[i + 2] - 2 * x[i + 1] + 2 * x[i - 1 + n] - x[i - 2 + n]) * 0.5 * tpoints;

    i = 1;
    dst[i] = -6 * x[i] * (x[i + 1] - x[i - 1]) * 0.5 * points
             - (x[i + 2] - 2 * x[i + 1] + 2 * x[i - 1] - x[i - 2 + n]) * 0.5 * tpoints;

#pragma omp parallel for shared(dst) private(i)
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