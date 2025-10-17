import os
import subprocess
import joblib
import pytest

def test_training_creates_model(tmp_path):
    out_csv = tmp_path / "train.csv"
    # generate small dataset
    cmd = ["python", "data/generate_synthetic_data.py", "--out", str(out_csv), "--n-stores", "1", "--n-products", "1", "--periods", "60"]
    subprocess.check_call(cmd)
    # train
    out_dir = tmp_path / "models"
    cmd2 = ["python", "src/train.py", "--data-path", str(out_csv), "--model", "linear", "--output-dir", str(out_dir)]
    subprocess.check_call(cmd2)
    model_path = out_dir / "model.joblib"
    scaler_path = out_dir / "scaler.joblib"
    assert model_path.exists()
    assert scaler_path.exists()
    # try loading
    m = joblib.load(str(model_path))
    s = joblib.load(str(scaler_path))
    assert m is not None
    assert s is not None