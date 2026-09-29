"""
Experiment 6: Model Registry Pipeline

Fake & Real News Detection

This pipeline:
1. Registers a trained model
2. Displays registered model versions
3. Demonstrates model promotion
4. Demonstrates rollback

Run from project root:

python pipelines/run_model_registry.py
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT MODULES
# ============================================================

from register_model import main as register_model

from compare_models import main as compare_models

from rollback_model import main as rollback_model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 6: MODEL REGISTRY PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\n[STEP 1] Registering model..."
    )

    register_model()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n\n[STEP 2] Comparing model versions..."
    )

    compare_models()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n\n[STEP 3] Model lifecycle management..."
    )

    rollback_model()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "EXPERIMENT 6 PIPELINE COMPLETED"
    )
    print("=" * 70)

    print(
        "\nModel registry workflow completed:"
    )

    print(
        "1. Model registered"
    )

    print(
        "2. Model versions inspected"
    )

    print(
        "3. Model promoted using 'champion' alias"
    )

    print(
        "4. Rollback demonstrated"
    )


if __name__ == "__main__":
    main()