import random
import os
import math
import argparse

def generate_data_with_ranges(filename, num_vectors, length_per_vector,
                              overlap_ratio,
                              distribution_type="range_overlap",
                              full_value_range=(0, 1000000)
                             ):

    print(f"Generating data for: {filename}...")
    print(f"  Vectors: {num_vectors}, Target Length: {length_per_vector}, Overlap Ratio: {overlap_ratio}")
    print(f"  Distribution Type: {distribution_type}")

    full_min, full_max = full_value_range

    if distribution_type == "range_overlap":
        overlap_max = int(length_per_vector * overlap_ratio)
        overlap_range = (full_min, overlap_max)
        print(f"  Overlap range: {overlap_range}")

        overlap_base_set = set(range(overlap_range[0], overlap_range[1]))
        overlap_base = sorted(list(overlap_base_set))

        if len(overlap_base) < int(length_per_vector * overlap_ratio):
            print(f"  Warning: Not enough possible overlap values. Have {len(overlap_base)}, need {int(length_per_vector * overlap_ratio)}")

    elif distribution_type == "uniform":
        overlap_range = full_value_range
        print(f"  Overlap range: {overlap_range} (full range)")

        overlap_size = int(length_per_vector * overlap_ratio)
        overlap_base_set = set()
        attempts = 0
        max_attempts = overlap_size * 10 + 1000

        while len(overlap_base_set) < overlap_size and attempts < max_attempts:
            val = random.randint(full_min, full_max - 1)
            overlap_base_set.add(val)
            attempts += 1

        overlap_base = sorted(list(overlap_base_set))

        if len(overlap_base) < overlap_size:
            print(f"  Warning: Could only generate {len(overlap_base)}/{overlap_size} overlap elements after {max_attempts} attempts.")

    else:
        print(f"Error: Unknown distribution type '{distribution_type}'")
        return

    with open(filename, 'w') as f:
        for i in range(num_vectors):
            num_overlap_target = int(length_per_vector * overlap_ratio)

            if len(overlap_base) > num_overlap_target:
                overlap_elements = sorted(random.sample(overlap_base, num_overlap_target))
            else:
                overlap_elements = overlap_base.copy()

            num_unique_needed = length_per_vector - len(overlap_elements)

            unique_elements = set()
            attempts = 0
            max_attempts = num_unique_needed * 10 + 1000

            while len(unique_elements) < num_unique_needed and attempts < max_attempts:
                val = random.randint(full_min, full_max - 1)
                if val not in overlap_base_set and val not in unique_elements:
                    unique_elements.add(val)
                attempts += 1

            if len(unique_elements) < num_unique_needed:
                print(f"  Warning: Vector {i}: Could only generate {len(unique_elements)}/{num_unique_needed} unique elements after {max_attempts} attempts.")

            full_vec = sorted(list(overlap_elements) + list(unique_elements))

            f.write(" ".join(map(str, full_vec)) + "\n")
            if i == 0:
                print(f"  Vector 0 actual size: {len(full_vec)} (Target: {length_per_vector}, Overlap target: {num_overlap_target}, Unique target: {num_unique_needed}, Unique generated: {len(unique_elements)})")

    print(f"Finished: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector data with controlled overlap in specific range or uniform distribution.")
    parser.add_argument("--length", type=int, required=True, help="Target number of elements per vector.")
    parser.add_argument("--overlap-ratio", type=float, required=True, help="Target overlap ratio (e.g., 0.4 for 40%).")
    parser.add_argument("--distribution", type=str, default="range_overlap", choices=['range_overlap', 'uniform'],
                       help="Distribution type: 'range_overlap' (overlap in range [0, length*ratio)) or 'uniform' (overlap from full range)")
    parser.add_argument("--num-vectors", type=int, default=3, help="Number of vectors to generate (default: 3).")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the output file (default: data).")
    parser.add_argument("--value-max", type=int, default=1000000, help="Maximum value for generated numbers (exclusive, default: 1,000,000).")

    args = parser.parse_args()

    if not 0.0 <= args.overlap_ratio <= 1.0:
        print("Error: Overlap ratio must be between 0.0 and 1.0.")
        exit(1)

    FULL_VALUE_RANGE = (0, args.value_max)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.distribution == "range_overlap":
        filename_suffix = f"skew_len{args.length}.txt"
    else:
        filename_suffix = f"uniform_len{args.length}.txt"

    output_filename = os.path.join(args.output_dir, f"{filename_suffix}")

    generate_data_with_ranges(output_filename,
                              args.num_vectors,
                              args.length,
                              args.overlap_ratio,
                              distribution_type=args.distribution,
                              full_value_range=FULL_VALUE_RANGE)

    print(f"\nData generation complete. File saved to: {output_filename}")