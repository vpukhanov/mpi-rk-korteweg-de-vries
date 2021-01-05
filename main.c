#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "rk.h"

FILE *foutput;

void print_report(double *, double, int);

/* main [points factor=10] [omp threads count=4] */
int main(int argc, char **argv) {
    int points_factor = argc >= 2 ? atoi(argv[1]) : 10;
    int threads_count = argc >= 3 ? atoi(argv[2]) : 4;

    int steps_num = 50 * points_factor;

    double start = 0;
    double end = 10;
    double step = 0.001;

    double *x = (double *) malloc(steps_num * sizeof(double));

    double t1, t2;
    int i;

    omp_set_num_threads(threads_count);
    t1 = omp_get_wtime();

#pragma omp parallel for shared(x, steps_num) private(i)
    for (i = 0; i < steps_num; i++) {
        double ox = i * 0.1;
        double oxcosh = cosh(ox - 25);
        x[i] = 2 / (oxcosh * oxcosh);
    }

    foutput = fopen("output.txt", "w");
    runge_kutta(x, step, steps_num, start, end, print_report);
    fclose(foutput);

    t2 = omp_get_wtime();
    printf("Total time for %d threads: %.5f", threads_count, t2 - t1);

    free(x);

    return 0;
}

void print_report(double *arr, double t, int num) {
    int i;

    fprintf(foutput, "%.5f\n", t);
    for (i = 0; i < num; i++) {
        fprintf(foutput, "%.5f ", arr[i]);
    }

    fprintf(foutput, "\n");
}