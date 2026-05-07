from ultralytics import YOLO




class VisionDetector:
    """
    Class VisionDetector uses YOLO model to detect frame input
    """

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

        
    def detect(self, frame):
        results = self.model(frame)
        boxes = results[0].boxes # first results framed boxes 
        annotated_frame = results[0].plot()

        return results, boxes, annotated_frame
