"""
Experiment 6: Model Version Comparison

Displays all registered versions of the
Fake & Real News SVM model.

Run:

python src/compare_models.py
"""

import mlflow


REGISTERED_MODEL_NAME = (
    "Fake_Real_News_SVM"
)


def main():

    print("=" * 70)
    print("MODEL VERSION COMPARISON")
    print("=" * 70)

    client = mlflow.MlflowClient()

    try:

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

    except Exception as error:

        print(
            f"\nUnable to retrieve model versions:\n"
            f"{error}"
        )

        return

    if not versions:

        print(
            "\nNo registered model versions found."
        )

        return

    print(
        f"\nRegistered Model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"\nTotal versions: "
        f"{len(versions)}"
    )

    print("\n" + "-" * 70)

    for version in versions:

        print(
            f"\nModel Version : "
            f"{version.version}"
        )

        print(
            f"Run ID        : "
            f"{version.run_id}"
        )

        print(
            f"Status        : "
            f"{version.status}"
        )

        print(
            f"Created       : "
            f"{version.creation_timestamp}"
        )

        print(
            f"Source        : "
            f"{version.source}"
        )

        print("-" * 70)


if __name__ == "__main__":
    main()