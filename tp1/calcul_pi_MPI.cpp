# include <chrono>
# include <random>
# include <cstdlib>
# include <sstream>
# include <string>
# include <fstream>
# include <iostream>
# include <iomanip>
# include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"

double approximate_pi(unsigned long nbSamples) 
{
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = beginning.time_since_epoch();
    unsigned seed = d.count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    unsigned long nbDarts = 0;
    for (unsigned sample = 0; sample < nbSamples; ++sample) {
        double x = distribution(generator);
        double y = distribution(generator);
        if (x * x + y * y <= 1) nbDarts++;
    }

    return double(nbDarts) / double(nbSamples);
}

int main(int nargs, char* argv[]) 
{
    MPI_Init(&nargs, &argv);
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);

    int nbp, rank;
    MPI_Comm_size(globComm, &nbp);
    MPI_Comm_rank(globComm, &rank); 

    std::stringstream fileName;
    fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
    std::ofstream output(fileName.str().c_str());

    unsigned long totalSamples = 100000000; 
    unsigned long localSamples = totalSamples / nbp; 

    auto start = std::chrono::high_resolution_clock::now();

    double local_pi = approximate_pi(localSamples);
    double global_pi = 0.0;

    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, globComm);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        global_pi = 4 * (global_pi / nbp);
        std::cout << "Estimated Pi: " << global_pi << std::endl;
        std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;
    }

    output.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}
