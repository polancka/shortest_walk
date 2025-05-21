import sys
import re

def calculate_total_execution_time(log_file='log.txt'):
    total_time = 0.0
    time_pattern = re.compile(r"Execution time: ([\d.]+) seconds")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = time_pattern.search(line)
            if match:
                total_time += float(match.group(1))

    print(f"Total execution time: {total_time:.6f} seconds")

if __name__ == "__main__":
    calculate_total_execution_time(sys.argv[1])