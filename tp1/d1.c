# include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
# include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, token;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("Ce programme doit être exécuté avec 2 processus.\n");
        }
        MPI_Finalize();
        return 1;
    }

    start_time = MPI_Wtime(); 

    if (rank == 0) {
        token = 42; 
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Processus 0 a envoyé le jeton %d à processus 1.\n", token);
    } else if (rank == 1) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 1 a reçu le jeton %d de processus 0.\n", token);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Temps total de diffusion: %f secondes.\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
