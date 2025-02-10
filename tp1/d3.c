#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, token;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 8) {
        if (rank == 0) {
            printf("Ce programme doit être exécuté avec 8 processus.\n");
        }
        MPI_Finalize();
        return 1;
    }

    start_time = MPI_Wtime(); 

    if (rank == 0) {
        token = 42;
        int targets[] = {1, 2, 4};
        for (int i = 0; i < 3; i++) {
            MPI_Send(&token, 1, MPI_INT, targets[i], 0, MPI_COMM_WORLD);
        }
        printf("Processus 0 a envoyé le jeton %d aux processus 1, 2 et 4.\n", token);
    } 
    if (rank == 1) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 1 a reçu le jeton %d de processus 0.\n", token);
        MPI_Send(&token, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
        MPI_Send(&token, 1, MPI_INT, 5, 0, MPI_COMM_WORLD);
    }
    if (rank == 2) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 2 a reçu le jeton %d de processus 0.\n", token);
        MPI_Send(&token, 1, MPI_INT, 6, 0, MPI_COMM_WORLD);
    }
    if (rank == 3) {
        MPI_Recv(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 3 a reçu le jeton %d de processus 1.\n", token);
        MPI_Send(&token, 1, MPI_INT, 7, 0, MPI_COMM_WORLD);
    }
    if (rank == 4) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 4 a reçu le jeton %d de processus 0.\n", token);
    }
    if (rank == 5) {
        MPI_Recv(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 5 a reçu le jeton %d de processus 4.\n", token);
    }
    if (rank == 6) {
        MPI_Recv(&token, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 6 a reçu le jeton %d de processus 2.\n", token);
    }
    if (rank == 7) {
        MPI_Recv(&token, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus 7 a reçu le jeton %d de processus 6.\n", token);
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    end_time = MPI_Wtime(); 

    if (rank == 0) {
        printf("Temps total de diffusion: %f secondes.\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
