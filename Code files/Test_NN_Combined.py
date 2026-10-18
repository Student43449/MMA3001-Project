"""Unit tests for the combined golf-shot neural network.

These tests check four important behaviours:
1. The target columns are not used as input features (preventing data leakage).
2. The network produces two predictions per shot.
3. The scaler learns statistics only from training data.
4. One training step updates network weights and produces a finite loss.

The NN_Combined.py script trains for 150 epochs and displays plots when run.
To avoid doing that during testing, this file reads its Python syntax tree and
loads only the GolfNeuralNet class, without executing the training script.

Run from the repository root:
    python -m pytest "Code files/Test_NN_Combined.py" -v

Requires: pytest, torch, numpy, scikit-learn.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler


# Locate the real script regardless of the directory pytest was started from.
SCRIPT_PATH = Path(__file__).resolve().with_name("NN_Combined.py")


@pytest.fixture(scope="module")
def script_tree():
    """Parse the source without running its training or plotting code."""
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))


def assigned_value(tree, variable_name):
    """Return a literal list assigned to a top-level variable in the script."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    pytest.fail(f"Could not find a simple assignment for {variable_name!r}")


@pytest.fixture(scope="module")
def network_class(script_tree):
    """Load only GolfNeuralNet, avoiding the script's top-level training loop."""
    class_node = next(
        (node for node in script_tree.body
         if isinstance(node, ast.ClassDef) and node.name == "GolfNeuralNet"),
        None,
    )
    assert class_node is not None, "GolfNeuralNet class is missing"
    # Compile the class definition in isolation; nn must exist in its namespace.
    namespace = {"nn": torch.nn}
    isolated_module = ast.fix_missing_locations(
        ast.Module(body=[class_node], type_ignores=[])
    )
    exec(compile(isolated_module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["GolfNeuralNet"]


def test_features_exclude_targets(script_tree):
    """The measured outcomes must never be passed to the network as inputs."""
    features = assigned_value(script_tree, "features")
    targets = assigned_value(script_tree, "targets")

    assert targets == ["Side Total", "Total"]
    assert len(features) == 15
    assert len(features) == len(set(features)), "Input features contain duplicates"
    assert not set(features).intersection(targets), "Target leakage into input features"


def test_network_output_dimensions(network_class):
    """Five dummy golf shots should produce five pairs of predictions."""
    model = network_class(input_dim=15)
    dummy_shots = torch.randn(5, 15)

    model.eval()
    with torch.no_grad():
        predictions = model(dummy_shots)

    assert predictions.shape == (5, 2)
    assert torch.isfinite(predictions).all()


def test_scaler_uses_training_data_only(script_tree):
    """Check the script's scaler calls and demonstrate their expected behaviour."""
    # Inspect actual source calls: fitting must use X_train, not X_val.
    calls = [
        node for node in ast.walk(script_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scaler"
    ]
    fitting_calls = [
        call for call in calls if call.func.attr in ("fit", "fit_transform")
    ]
    assert len(fitting_calls) == 1, "Scaler should be fitted exactly once"
    assert len(fitting_calls[0].args) >= 1
    assert isinstance(fitting_calls[0].args[0], ast.Name)
    assert fitting_calls[0].args[0].id == "X_train"

    validation_transforms = [
        call for call in calls
        if call.func.attr == "transform"
        and call.args and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "X_val"
    ]
    assert validation_transforms, "Validation features must use scaler.transform"

    # Small synthetic example: validation data must not change training means.
    train = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    validation = np.array([[100.0, 200.0]])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    training_means = scaler.mean_.copy()
    scaled_validation = scaler.transform(validation)

    np.testing.assert_allclose(scaler.mean_, training_means)
    np.testing.assert_allclose(scaled_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
    assert np.isfinite(scaled_validation).all()


def test_one_training_step_updates_weights(network_class):
    """A single optimisation step should compute a finite loss and change weights."""
    torch.manual_seed(42)  # Make the test repeatable.
    model = network_class(input_dim=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Synthetic inputs/targets keep the test fast and independent of CSV files.
    inputs = torch.randn(8, 15)
    targets = torch.randn(8, 2)
    original_weights = [parameter.detach().clone() for parameter in model.parameters()]

    model.train()
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    assert torch.isfinite(loss), "Training loss must be finite"

    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_weights, model.parameters())
    ), "No network parameters changed during training"
