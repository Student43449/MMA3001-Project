import os
import subprocess
import sys

def run_pipeline():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_script = os.path.join(code_dir, 'Data_processing.py')
    nn_combined_script = os.path.join(code_dir, 'NN_Combined.py')

    print("=== Step 1: Running Data Processing & Dataset Splitting ===")
    result_dp = subprocess.run([sys.executable, data_processing_script], capture_output=True, text=True)
    print(result_dp.stdout)
    if result_dp.returncode != 0:
        print("Data processing failed:", result_dp.stderr)
        sys.exit(1)

    print("=== Step 2: Running NN_Combined Training and Displaying Outputs ===")
    result_nn = subprocess.run([sys.executable, nn_combined_script], capture_output=True, text=True)
    print(result_nn.stdout)
    if result_nn.returncode != 0:
        print("NN_Combined training failed:", result_nn.stderr)
        sys.exit(1)

if __name__ == '__main__':
    run_pipeline()
