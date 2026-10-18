"""Run the main golf-shot machine-learning workflow.

Run this file to prepare data, evaluate linear regression baselines, and train
and evaluate the combined neural network. Each step runs in its own Python
process so scripts can also be executed individually.

Example (from the repository root):
    python Main.py
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE_FOLDER = ROOT / "Code files"


def run_script(filename):
    """Run a project script and stop if it exits with an error.

    Args:
        filename (str): Filename within the Code files directory.
    """
    script = CODE_FOLDER / filename
    if not script.is_file():
        raise FileNotFoundError(f"Cannot find project script: {script}")

    print(f"\nRunning {filename}...", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main():
    """Run data preparation, baseline regression, and combined neural network."""
    run_script("Data_processing.py")
    run_script("linear_regression_side_total.py")
    run_script("NN_Combined.py")


if __name__ == "__main__":
    main()
