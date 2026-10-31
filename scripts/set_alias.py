import argparse
import os

from mlflow import MlflowClient


parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("version")
parser.add_argument("alias")
args = parser.parse_args()

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
client.set_registered_model_alias(args.name, args.alias, args.version)
print(f"{args.name}@{args.alias} -> v{args.version}")
