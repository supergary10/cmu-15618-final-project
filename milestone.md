# <center>Parallel Worst-Case Optimal Join</center>
<center>Authors: Dapeng Gao, Yifan Guang</center>

## Sumamry
We have implemented the serial and parallel versions of a naive worst-case optimal join algorithm. We have tested various platforms and implementations of WCOJ. It generally keeps align with the original timeline. However, as our development work progressed, we realized that achieving all of the original desired goals was nearly impossible. We will focus on the parallel aspects of the optimization while the optimization of the algorithm itself we will no longer consider.

The details of the progress and the changes of the project are as follows.
## Progress
### Implementation of Benchmark Infrastructures
We have implemented the benchmark infrastructures for the testing the computation time of the naive WCOJ algorithm. We will use the benchmark infrastructures to test the performance of our parallel implementation in the future.

### Implementation of the Testcases Generator
We are focusing on the join functions for the in memory database. The testcases generator can produce some numbers to construct the input vectors that act as the input relations for the join functions. There is also a data loader that can read the generated testcases into vectors. The testcases generator can generate the input relations with different sizes and distributions. We want to generate different kinds of testcases and test the performance of our parallel algorithms on them.

### Implementation of the Serial Version of naive WCOJ
The naive version of the WCOJ algorithm has been implemented. It will go through the input relations and find the next tuple that satisfies the join condition. As our implementation is unary, it will only find the intersection of all the input vectors and return the result. These elements represent the IDs of a database and simulate the join condition. The naive WCOJ algorithm is a simple implementation of the WCOJ algorithm. It is not efficient for large input relations, but it is easy to understand and implement. We have tested the naive WCOJ algorithm on the benchmark infrastructures and it works as expected. The performance of the naive WCOJ algorithm is acceptable for small input relations, but it becomes slower for larger input relations. We will do further performance evaluation on more testcases.
Note that we are using ordered testcase now. Otherwise, we need to sort the input vector. This limitation is not a part of parallel problem.

### Implementation of the Parallel Version of naive WCOJ
We have started implementing the parallel version of the naive WCOJ algorithm. Currently, we have achieved parallel the outer most vector and split it into multiple parts into multiple threads to run the naive WCOJ algorithm. We utilize OpenMP to achieve the parallelism. Also, we will do more performance evaluations and parallel optimizations.

### Implementation of the Leapfrog Join
We have also implemented the leapfrog join algorithm, which is an efficient implementation of the WCOJ algorithm. Compared to the naive WCOJ algorithm, the leapfrog join algorithm is more efficient at searching for the next tuple particularly when the input relations are large. We have implemented the leapfrog join algorithm in both serial and parallel versions and tested them on the benchmark infrastructures. The results show that the leapfrog join algorithm is significantly faster than the naive WCOJ algorithm, especially for large input relations. However, it is not a part of the parallel problem we are focusing on, so we may not continue focus on these kind of optimizations.

## Problems
- The major problems is that we want and we are supposed to focus on the parallel problems of the WCOJ algorithm. However, there are a lot of other problems that need to be implemented, like how to maintain the data structure and do the join on non-unary relations. Therefore, we will only focus on some narrow and specific problems of the WCOJ algorithm to achieve the parallelism optimization. 
- Another thing is that we need to create the testacases that are suitable for our problem. They may not be comprehensive enough or have other problems for the in memory running and testing the join algorithm.
- Currently the GHC machines work well for our project. We may need to use the PSC for further development and performance evaluation. This is an uncertainty.

## Updated Goals
Considering the progress we have made so far and the problems we have encountered, we have updated our goals for the project. The updated goals are as follows:
- Implement sequential WCOJ as baseline (already done)
- Implement simple parallel WCOJ with top-level parallelism (already done)
- Implement job stealing and SPMD for the simple parallel WCOJ (Will explore in the following weeks)
- Implement advanced parallel WCOJ, like parallelizing the inner loop (Maybe not possible if involes too much database works)
- Try different partition strategies (Partially done)
- Analyze load balancing, total work, and speedup on multicore CPUs (In progress)
- Generate more testcases for performance evaluation (In progress)

## Updated Timeline
| Date | Task                                                         |
|------|--------------------------------------------------------------|
| 4.16 | Generate more testcases for performance testing              |
| 4.20 | Performance evaluation for basic parallelism and develop     |
| 4.23 | Continue develop advanced parallel optimization              |
| 4.27 | Performance evaluation and final report writing              |
| 4.29 | Poster session                                               |

## Poster Session
We plan to present a graph of our project during the poster session. The graph will include the following information:
- The performance of the naive WCOJ algorithm
- The performance of the parallel naive WCOJ algorithm
- The performance of the advanced parallel naive WCOJ algorithm
- Some other analysis, like cache misses and speedup.
- The graph that illustrates how the parallel algorithm works
- Brief introduction of the background of the project
- Miscellaneous, including experiments setup, testcases, and other information