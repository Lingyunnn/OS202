# Parallel Bucket Sort Using MPI

The code utilizes **MPI (Message Passing Interface)** to perform **parallel bucket sort**. The execution process is as follows:

1. Each process generates data, computes its own **quantiles**, and sends them to **process 0**.
2. **Process 0** calculates the **global quantiles** and broadcasts them to all processes.
3. Each process distributes its data into **buckets** based on the quantiles.
4. Processes **exchange data**, ensuring that each process collects the data for its assigned bucket.
5. Each process **sorts** its bucket data.
6. **Process 0** gathers all the sorted data and outputs the final result.
