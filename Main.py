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
    """Execute one project script and stream its printed output and errors."""
    script = CODE_FOLDER / filename
    if not script.is_file():
        raise FileNotFoundError(f"Cannot find project script: {script}")

    print(f"\n=== Running {filename} ===", flush=True)
    # No output capture: results and errors appear directly in Colab.
    subprocess.run([sys.executable, "-u", str(script)], cwd=ROOT, check=True)


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

    run_script("Data_processing.py")
    run_script("linear_regression_side_total.py")
    run_script("NN_Combined.py")
    show_graphs()


if __name__ == "__main__":
    main()
