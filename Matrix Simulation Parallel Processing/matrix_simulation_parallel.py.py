from multiprocessing import Process, shared_memory
import argparse
import os
import math

# Helper Functions
def is_prime(num):
    """Check if a number is prime."""
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def is_power_of_2(num):
    """Check if a number is a power of 2."""
    return num > 0 and (num & (num - 1)) == 0

# Matrix Utilities
def get_neighbors(matrix, rows, cols, x, y):
    """Retrieve the values of neighboring cells."""
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append(matrix[nx * cols + ny])
    return neighbors

def compute_neighbor_sum(matrix, rows, cols, x, y):
    """Compute the weighted sum of neighbors."""
    neighbor_values = {
        'O': 2,
        'o': 1,
        '.': 0,
        'x': -1,
        'X': -2
    }
    neighbors = get_neighbors(matrix, rows, cols, x, y)
    return sum(neighbor_values.get(chr(neighbor), 0) for neighbor in neighbors)

def process_cell(current_value, neighbor_sum):
    """Apply rules to determine the next state of a cell."""
    current_value = chr(current_value)  # Convert from int to char
    if current_value == 'O':  # Healthy O Cell
        if is_power_of_2(neighbor_sum):
            return ord('.')
        elif neighbor_sum < 10:
            return ord('o')
        return ord('O')
    elif current_value == 'o':  # Weakened O Cell
        if neighbor_sum <= 0:
            return ord('.')
        elif neighbor_sum >= 8:
            return ord('O')
        return ord('o')
    elif current_value == '.':  # Dead Cell
        if is_prime(neighbor_sum):
            return ord('o')
        elif is_prime(abs(neighbor_sum)):
            return ord('x')
        return ord('.')
    elif current_value == 'x':  # Weakened X Cell
        if neighbor_sum >= 1:
            return ord('.')
        elif neighbor_sum <= -8:
            return ord('X')
        return ord('x')
    elif current_value == 'X':  # Healthy X Cell
        if is_power_of_2(abs(neighbor_sum)):
            return ord('.')
        elif neighbor_sum > -10:
            return ord('x')
        return ord('X')
    return ord(current_value)

# Worker Function
def worker_process(start_row, end_row, rows, cols, shared_matrix_name, shared_next_name):
    """Worker process to compute a range of rows."""
    shm_matrix = shared_memory.SharedMemory(name=shared_matrix_name)
    shm_next = shared_memory.SharedMemory(name=shared_next_name)

    matrix = shm_matrix.buf
    next_matrix = shm_next.buf

    for x in range(start_row, end_row):
        for y in range(cols):
            neighbor_sum = compute_neighbor_sum(matrix, rows, cols, x, y)
            next_matrix[x * cols + y] = process_cell(matrix[x * cols + y], neighbor_sum)

    shm_matrix.close()
    shm_next.close()

# Parallel Matrix Processing
def simulate_parallel(matrix, rows, cols, iterations, num_processes):
    """Simulate matrix evolution using multiprocessing with shared memory."""
    shm_matrix = shared_memory.SharedMemory(create=True, size=rows * cols)
    shm_next = shared_memory.SharedMemory(create=True, size=rows * cols)

    shm_matrix.buf[:rows * cols] = bytes(matrix)

    for _ in range(iterations):
        processes = []
        rows_per_process = rows // num_processes
        for i in range(num_processes):
            start_row = i * rows_per_process
            end_row = (i + 1) * rows_per_process if i != num_processes - 1 else rows
            p = Process(target=worker_process, args=(start_row, end_row, rows, cols, shm_matrix.name, shm_next.name))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Swap shared memory buffers for the next iteration
        shm_matrix.buf[:rows * cols] = shm_next.buf[:rows * cols]

    result = list(shm_matrix.buf[:rows * cols])

    shm_matrix.close()
    shm_matrix.unlink()
    shm_next.close()
    shm_next.unlink()

    return result

# File I/O
def read_matrix(file_path):
    """Read the matrix from a file."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    rows = len(lines)
    cols = len(lines[0])
    flat_matrix = [ord(cell) for line in lines for cell in line]
    return flat_matrix, rows, cols

def write_matrix(matrix, rows, cols, file_path):
    """Write the matrix to a file."""
    with open(file_path, 'w') as f:
        for i in range(rows):
            f.write(''.join(chr(matrix[i * cols + j]) for j in range(cols)) + '\n')

# Main Function


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Matrix Simulation Program")
    parser.add_argument('-i', required=True, help='Input file path')
    parser.add_argument('-o', required=True, help='Output file path')
    parser.add_argument('-p', type=int, default=1, help='Number of processes (default: 1)')
    args = parser.parse_args()

    # Validate the number of processes
    if args.p <= 0:
        print("Error: Number of processes (-p) must be a positive integer.")
        return

    input_file = args.i
    output_file = args.o
    num_processes = args.p

    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    # Read the matrix
    matrix, rows, cols = read_matrix(input_file)
    print(f"Processing matrix for 100 iterations with {num_processes} processes...")

    # Process the matrix
    final_matrix = simulate_parallel(matrix, rows, cols, iterations=100, num_processes=num_processes)

    # Write the processed matrix to the output file
    write_matrix(final_matrix, rows, cols, output_file)
    print(f"Final matrix written to '{output_file}'")

if __name__ == "__main__":
    main()

#-i "C:\Users\ashso\Documents\six_by_six_v2 (1)\time_step_0.dat" -o "C:\Users\ashso\Documents\output\time_step_100.dat" -p 20
