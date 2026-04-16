from roboflow import Roboflow

rf = Roboflow(api_key="2CmzVG8usizmcU8crCA5")

project = rf.workspace("eagle-eye").project("basketball-1zhpe")
version = project.version(1)

dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)