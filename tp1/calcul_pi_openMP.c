#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TOTAL_SAMPLES 100000000  

double approximate_pi(unsigned long nbSamples) {
    unsigned long nbDarts = 0;

    #pragma omp parallel 
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();
        srand(seed); 
        unsigned long local_nbDarts = 0;

        #pragma omp for
        for (unsigned long i = 0; i < nbSamples; i++) {
            double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand() / RAND_MAX * 2.0 - 1.0;

            if (x * x + y * y <= 1) local_nbDarts++;
        }

        #pragma omp atomic
        nbDarts += local_nbDarts;
    }

    return 4.0 * (double)nbDarts / (double)nbSamples;
}

int main() {
    double pi;
    double start_time, end_time;

    start_time = omp_get_wtime();
    pi = approximate_pi(TOTAL_SAMPLES);
    end_time = omp_get_wtime();

    printf("Estimated Pi: %f\n", pi);
    printf("Elapsed Time: %f seconds\n", end_time - start_time);

    return 0;
}
