import argparse
from tqdm import tqdm

def count_periods_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            period_count = 0
            file_lines = file.readlines()
            for line in file_lines:
                period_count += line.count('.')
        return period_count
    except FileNotFoundError:
        print("File not found.")
        return None

def main(file_path, total_runs):
    last_run_periods_count = 0
    print(count_periods_in_file(file_path))
    for i_value in tqdm(range(count_periods_in_file(file_path), total_runs)):
        while last_run_periods_count == count_periods_in_file(file_path):
            pass
        
        last_run_periods_count = count_periods_in_file(file_path)
        i_value = last_run_periods_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count periods in a file.")
    parser.add_argument("file", help="Path to the file to be processed.")
    parser.add_argument("total_runs", help="Number of runs in each process")
    args = parser.parse_args()
    filename = str(args.file) + "/progress_file.log"
    
    print(filename)
    main(filename, int(args.total_runs))
