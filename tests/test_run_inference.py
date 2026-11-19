import pytest
from pathlib import Path
from spot_detection_toolkit.run_inference import run_inference

@pytest.mark.parametrize("tool", ["piscis", "ufish", "spotiflow", "log"])
def test_run_inference(tool, tmp_path):
    input_dir = Path("tests/resources")
    results_dir = run_inference(
        tool,
        input_dir=input_dir,
        output_root=tmp_path,
        mode="2d",
    )
    assert (Path(results_dir).exists())
    assert any((Path(results_dir).iterdir()))
