# <center>Parallel Worst-Case Optimal Join</center>
<center>Authors: Dapeng Gao, Yifan Guang</center>

## Sumamry
We are going to investigate how to effectively parallelize Worst-Case Optimal Join (WCOJ) algorithms on multicore machines. We will try to optimize join performance by exploring full multi-variable partitioning, eager indexing (CoCo), and evaluating performance bottlenecks such as load skew and redundant computation.

## URL
[https://supergary10.github.io/cmu-15618-final-project/](https://supergary10.github.io/cmu-15618-final-project/)

[Milestone](https://supergary10.github.io/cmu-15618-final-project/milestone)

## Background
Worst-Case Optimal Join (WCOJ) algorithms evaluate multiway joins by iterating over join variables one at a time and performing multi-relation intersections. These algorithms are particularly effective for cyclic queries (like triangle queries), where traditional binary joins underperform.

Despite their theoretical efficiency, parallel WCOJ implementations remain difficult in practice. Most systems only parallelize the top-level variable. Although simple, this approach suffers from:
- Workload skew: some variables may have significantly more values or larger fanout.
- Redundant computation: shared relations are scanned repeatedly.
- Poor scaling: due to contention and lack of cache-aware partitioning.

Our project explores these challenges empirically by starting from a minimal baseline and optimizing incrementally.

## Challenge
There are several reasons this project is challenging from a parallel systems perspective:

- Workload characteristics: WCOJ performs nested loop joins with multi-way intersections. These intersections have poor memory locality and non-uniform compute intensity.
- Load skew: Real-world datasets often exhibit non-balanced distributions; a naive strategy may give one thread significantly more work than others.
- Contention: Lazy construction of trie indexes across threads leads to locking overhead and performance degradation.
- Partition strategy: Determining the right number of partitions per variable is difficult. The choice impacts cache efficiency, memory usage, and balance.
- Redundant computation: Relations that are not partitioned are repeatedly accessed by all threads, increasing total work.

It is likely that there is no a possible optimal solution that can solve all the challenges. We will aim to learn how fine-grained control over workload distribution and index layout can improve parallel join performance.

## Resources
We will use the following resources to complete our project:
- GHC machines for development and testing
- C++ with OpenMP for parallelization
- Starter code of the basic WCOJ algorithm
- Some papers for reference (in the References section)
- Dataset for testing

## Goals and Deliverables

### Plan to Achieve
- Implement sequential WCOJ as baseline
- Implement simple parallel WCOJ with top-level parallelism
- Implement multi-variable partitioning and eager indexing
- Analyze load balancing, total work, and speedup on multicore CPUs

### Hope to Achieve
- Try different partition strategies
- Optimize cache efficiency like intersection caching
- Testing on a variety of real-world datasets

### Fallback Plan
- Limit join pattern to triangle queries only
- Use fixed partitioning parameters rather than dynamic share allocation (ignore the partition strategy)

## Choice of Platform
We will use GHC machines for development and testing. As it has a i7-9700 CPU with 8 cores and 16GB memory, it is suitable for our project. If this is not enough, we will use the PSC cluster.

We are familiar with C++, and we have experienced a lot of OpenMP in Assignment 3. CUDA utilized SIMD, which is not suitable for our project. The performance of MPI is not as good as OpenMP from the results of Assignment 3 and 4.

## Schedule

| Date | Task                                                         |
|------|--------------------------------------------------------------|
| 4.2  | Finish the basic sequential WCOJ algorithm                   |
| 4.9  | Implement the simple parallel WCOJ algorithm                 |
| 4.10 | Performance evaluation and optimization analysis             |
| 4.14 | Implement the advanced parallel WCOJ algorithm               |
| 4.15 | Performance evaluation and optimization analysis             |
| 4.26 | Finish all the optimizations and evaluation                  |
| 4.28 | Final report writing                                         |
| 4.29 | Poster session                                               |

## References

1. Wu, J., & Suciu, D. (2025). HoneyComb: A Parallel Worst-Case Optimal Join on Multicores. arXiv preprint arXiv:2502.06715.

2. Ngo, H. Q., Porat, E., Ré, C., & Rudra, A. (2018). Worst-case optimal join algorithms. Journal of the ACM (JACM), 65(3), 1-40.

3. SNAP Datasets: [https://snap.stanford.edu/data/](https://snap.stanford.edu/data/)
