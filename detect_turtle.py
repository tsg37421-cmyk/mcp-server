from ultralytics import YOLO


# Load a model

model = YOLO('C:/Users/J/Desktop/mcp-serversum_best.pt')  # load a custom model

#Run inference on the source
input_folder = "C:/Users/J/Desktop/mcp-server/input"
output_folder = "C:/Users/J/Desktop/mcp-server/output"
model.predict(source=input_folder, project=output_folder, conf=0.7, save_conf=True, save=True)
