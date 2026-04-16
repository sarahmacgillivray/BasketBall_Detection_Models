import yaml
import os

dataset_path = "basketball-1"  # update if needed
yaml_path = os.path.join(dataset_path, "data.yaml")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

data["path"] = os.path.abspath(dataset_path)

with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print("Fixed data.yaml path")