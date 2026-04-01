#!/usr/bin/env python3
"""
Predict properties with ALIGNN from stdin JSON.

Input JSON:
  {
    "structure": {"lattice": [[...]], "positions": [[...]], "species": [...]},
    "properties": ["bulk_modulus", "shear_modulus"],
    "device": "cpu",
    "model_dir": null
  }
"""

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Force CPU-only mode for DGL/ALIGNN to avoid CUDA version conflicts
# This must be set before importing DGL or ALIGNN
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="ALIGNN property predictor")
    parser.add_argument("--input", type=str, default=None, help="Path to JSON input file")
    args = parser.parse_args()

    if args.input:
        try:
            raw = Path(args.input).read_text().strip()
        except Exception as exc:
            print(json.dumps({"error": f"failed to read input: {exc}"}))
            return 1
    else:
        raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "empty input"}))
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid json: {exc}"}))
        return 1

    struct_data = payload.get("structure", {})
    lattice = struct_data.get("lattice")
    positions = struct_data.get("positions")
    species = struct_data.get("species")
    properties = payload.get("properties") or []

    if not lattice or not positions or not species:
        print(json.dumps({"error": "missing structure data"}))
        return 1

    def _run_prediction() -> dict:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            try:
                from pymatgen.core import Lattice, Structure
                from src.surrogate_models import ALIGNNPredictor

                structure = Structure(Lattice(lattice), species, positions)
                predictor = ALIGNNPredictor(
                    model_dir=payload.get("model_dir"),
                    device=payload.get("device", "cpu"),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictions = predictor.predict_multiple(structure, properties)

                result = {"predictions": predictions}

                # If any predictions are None, include debug info from stderr
                if any(v is None for v in predictions.values()):
                    stderr_content = err_buf.getvalue().strip()
                    if stderr_content:
                        result["debug"] = stderr_content[:4000]

                return result
            except Exception as exc:
                detail = (err_buf.getvalue() or out_buf.getvalue()).strip()
                result = {"error": str(exc)}
                if detail:
                    result["detail"] = detail[:4000]
                return result

    result = _run_prediction()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
