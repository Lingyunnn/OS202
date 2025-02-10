#include <stdio.h>
#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"

int main(int argc, char *argv[]) {
    int rank, size, token;

    // Initialiser MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // obtenir le rang du processus
    MPI_Comm_size(MPI_COMM_WORLD, &size); // obtenir le nombre total de processus

    // Initialiser le jeton
    if (rank == 0) {
        token = 1; // Le processus 0 initialise le jeton à 1
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // envoyer le jeton au processus suivant
    } else {
        // Recevoir le jeton, l'incrémenter, puis l'envoyer au suivant
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token++; // Incrémenter le jeton
        if (rank < size - 1) {
            MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        } else {
            // Le dernier processus renvoie le jeton au processus 0
            MPI_Send(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Le processus 0 affiche le jeton
    if (rank == 0) {
        MPI_Recv(&token, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Le jeton final est : %d\n", token); // Affiche le jeton
    }

    // Finaliser MPI
    MPI_Finalize();
    return 0;
}
