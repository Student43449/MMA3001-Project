"""Run the golf-shot prediction workflow and display its results.

Runs data preparation, linear regression and the combined neural network.
Each child script prints its progress and saves evaluation graphs as PNGs.

For inline graphs in Google Colab, run in a notebook cell:
    %run "/content/MMA3001-Project/Main.py"

Using !python also runs the workflow, but cannot render inline notebook
images from the separate Python process. PNGs are still saved to model outputs.
"""

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CODE_FOLDER = ROOT / "Code files"
OUTPUT_FOLDER = ROOT / "model outputs"

# List expected graph filenames in the order they should appear.
GRAPH_FILES = (
    "linear_regression_training.png",
    "linear_regression_validation.png",
    "nn_combined_side_total.png",
    "nn_combined_total.png",
)


def run_script(filename):
    """Execute a project script, stream its output, and return elapsed seconds."""
    script = CODE_FOLDER / filename
    if not script.is_file():
        raise FileNotFoundError(f"Cannot find project script: {script}")

    print(f"\n=== Running {filename} ===", flush=True)
    # No output capture: results and errors appear directly in Colab.
    # perf_counter measures elapsed (wall-clock) time, including startup and I/O.
    start = time.perf_counter()
    try:
        subprocess.run([sys.executable, "-u", str(script)], cwd=ROOT, check=True)
    finally:
        elapsed = time.perf_counter() - start
        print(f"Runtime for {filename}: {elapsed:.2f} seconds", flush=True)
    return elapsed


def show_graphs():
    """Display saved figures inline in IPython, or print their file paths."""
    print("\n=== Model evaluation graphs ===", flush=True)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        in_notebook = get_ipython() is not None and "ipykernel" in sys.modules
    except ImportError:
        in_notebook = False

    for filename in GRAPH_FILES:
        image_path = OUTPUT_FOLDER / filename
        if not image_path.is_file():
            print(f"Graph not found: {image_path}", flush=True)
            continue

        print(f"Graph: {filename}", flush=True)
        if in_notebook:
            # %run executes inside the notebook kernel, enabling inline display.
            display(Image(filename=str(image_path)))
        else:
            print(f"Saved at: {image_path}", flush=True)


def main():
    """Prepare data, evaluate both models, and show the saved graphs."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Delete only known old graph files, avoiding stale results after reruns.
    for filename in GRAPH_FILES:
        (OUTPUT_FOLDER / filename).unlink(missing_ok=True)

    # Store the elapsed time of each completed script for the final summary.
    runtimes = {}
    workflow_start = time.perf_counter()
    try:
        for filename in (
            "Data_processing.py",
            "linear_regression_side_total.py",
            "NN_Combined.py",
        ):
            runtimes[filename] = run_script(filename)
        show_graphs()
    finally:
        # Report completed scripts even if a later step fails.
        print("\n=== Runtime summary ===", flush=True)
        for filename, elapsed in runtimes.items():
            print(f"{filename}: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)", flush=True)
        print(f"Total workflow time: {time.perf_counter() - workflow_start:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
