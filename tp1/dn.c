# include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
# include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, d, token;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    d = 0;
    while ((1 << d) < size) d++;
    if ((1 << d) != size) {
        if (rank == 0) {
            printf("Le nombre de processus doit être une puissance de 2.\n");
        }
        MPI_Finalize();
        return 1;
    }

    start_time = MPI_Wtime();

    if (rank == 0) {
        token = 42;
        printf("Processus %d commence avec le jeton %d.\n", rank, token);
    }

    for (int i = 0; i < d; i++) {
        int partner = rank ^ (1 << i); 

        if (rank < partner) {
            MPI_Send(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            printf("Processus %d a envoyé le jeton %d à processus %d.\n", rank, token, partner);
        } else {
            MPI_Recv(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Processus %d a reçu le jeton %d de processus %d.\n", rank, token, partner);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Temps total de diffusion: %f secondes.\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
