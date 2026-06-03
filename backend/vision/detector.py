from ultralytics import YOLO
from vision.parser import DetectionParser
from datetime import datetime
import time



class VisionDetector:

    """
    Class VisionDetector uses YOLO model to detect frame input
    and converts detections into semantic objects.
    """


    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.parser = DetectionParser()

        
    def detect(self, frame, frame_id):
        start = time.time()
        results = self.model.track(
        frame,
        imgsz=320,
        conf=0.4,
        verbose=False
        )

        print(f"YOLO took {time.time() - start:.3f}s")

        result = results[0] # first frame 
        
        boxes = result.boxes # first results framed boxes 

        timestamp = datetime.now().isoformat()
        
        parsed_detections = self.parser.parse(boxes, self.model.names, frame_id, timestamp)
        
        annotated_frame = result.plot()

        return parsed_detections, annotated_frame

