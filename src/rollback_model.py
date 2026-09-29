"""
Experiment 6: Model Promotion and Rollback

Uses MLflow model aliases to demonstrate:

1. Model promotion
2. Model version selection
3. Rollback to a previous version

Run:

python src/rollback_model.py
"""

import mlflow


REGISTERED_MODEL_NAME = (
    "Fake_Real_News_SVM"
)


def main():

    print("=" * 70)
    print("MODEL PROMOTION / ROLLBACK")
    print("=" * 70)

    client = mlflow.MlflowClient()

    # --------------------------------------------------------
    # Get model versions
    # --------------------------------------------------------

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:

        print(
            "\nNo registered model versions found."
        )

        return

    # Sort by version number
    versions = sorted(
        versions,
        key=lambda x: int(x.version)
    )

    print(
        f"\nRegistered versions: "
        f"{len(versions)}"
    )

    for version in versions:

        print(
            f"Version {version.version}"
        )

    # --------------------------------------------------------
    # Latest version
    # --------------------------------------------------------

    latest_version = versions[-1]

    print(
        f"\nLatest version: "
        f"{latest_version.version}"
    )

    # --------------------------------------------------------
    # Promote latest model
    # --------------------------------------------------------

    print(
        "\nPromoting latest version "
        "as 'champion'..."
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        latest_version.version
    )

    print(
        f"Version {latest_version.version} "
        f"is now the champion model."
    )

    # --------------------------------------------------------
    # Rollback demonstration
    # --------------------------------------------------------

    if len(versions) >= 2:

        previous_version = versions[-2]

        print(
            f"\nRolling back to previous "
            f"version {previous_version.version}..."
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "champion",
            previous_version.version
        )

        print(
            f"Rollback completed."
        )

        print(
            f"Champion now points to "
            f"version {previous_version.version}."
        )

    else:

        print(
            "\nRollback demonstration requires "
            "at least two model versions."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL LIFECYCLE OPERATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()