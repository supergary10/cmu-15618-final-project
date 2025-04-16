import random
import os

def generate_case(filename, num_vectors, length_per_vector, overlap_ratio=0.01):
    # overlap base controls how likely an intersection is
    overlap_base = [random.randint(0, 10**9) for _ in range(int(length_per_vector * overlap_ratio))]
    with open(filename, 'w') as f:
        for _ in range(num_vectors):
            unique_part = [random.randint(0, 10**9) for _ in range(length_per_vector - len(overlap_base))]
            full_vec = list(set(overlap_base + unique_part))
            full_vec.sort()
            f.write(" ".join(map(str, full_vec)) + "\n")

os.makedirs("data", exist_ok=True)

# Previously small, medium, large
generate_case("data/test_small.txt", num_vectors=3, length_per_vector=10000, overlap_ratio = 0.05)
generate_case("data/test_medium.txt", num_vectors=3, length_per_vector=20000, overlap_ratio = 0.05)
generate_case("data/test_large.txt", num_vectors=3, length_per_vector=50000, overlap_ratio = 0.05)
