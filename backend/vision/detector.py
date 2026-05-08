from ultralytics import YOLO
from vision.parser import DetectionParser



class VisionDetector:
    """
    Class VisionDetector uses YOLO model to detect frame input
    and converts detections into semantic objects.
    """

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.parser = DetectionParser()

        
    def detect(self, frame):
        results = self.model(
        frame,
        imgsz=320,
        conf=0.4,
        verbose=False
        )

        result = results[0] # first frame 
        
        boxes = result.boxes # first results framed boxes 

        parsed_detections = self.parser.parse(boxes, self.model.names)
        
        annotated_frame = result.plot()

        return parsed_detections, annotated_frame

