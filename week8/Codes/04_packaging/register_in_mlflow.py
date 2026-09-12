"""
Register a packaged model in an MLflow Model Registry, including
metadata that lets future consumers trace provenance.

What this illustrates:
    - artifact logging (weights + model_card.json)
    - parameter and metric logging
    - registered model versioning (name -> version N)
    - stage promotion (None -> Staging -> Production)

Assumes a running MLflow tracking server (set MLFLOW_TRACKING_URI).

Usage:
    export MLFLOW_TRACKING_URI=http://mlflow.example.com
    python register_in_mlflow.py \
        --name resnet18_cifar10 \
        --artifacts ./exported \
        --val-acc 0.915
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--artifacts", required=True, help="directory of files to register")
    p.add_argument("--val-acc", type=float, required=True)
    p.add_argument("--train-loss", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--promote", choices=["Staging", "Production"], default=None)
    args = p.parse_args()

    with mlflow.start_run(run_name=f"register-{args.name}") as run:
        if args.epochs:
            mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("val_acc", args.val_acc)
        if args.train_loss is not None:
            mlflow.log_metric("train_loss", args.train_loss)

        artifact_dir = Path(args.artifacts)
        for f in artifact_dir.iterdir():
            mlflow.log_artifact(str(f))

        # Register the run as a new version of the named model.
        model_uri = f"runs:/{run.info.run_id}"
        result = mlflow.register_model(model_uri, args.name)
        print(f"Registered {args.name} v{result.version}")

        if args.promote:
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=args.name,
                version=result.version,
                stage=args.promote,
                archive_existing_versions=(args.promote == "Production"),
            )
            print(f"Transitioned v{result.version} -> {args.promote}")


if __name__ == "__main__":
    main()
