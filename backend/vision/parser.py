class DetectionParser:

    """
    Class DetectionParser converts YOLO output into semantic detection objects.
    """

    def parse(self, boxes, class_names):

        parsed_detections = []

        for i in range(len(boxes.cls)): #cls class labels for each box (for each labeled box)

            class_id = int(boxes.cls[i]) # make integer class labels per box 
            confidence = float(boxes.conf[i]) # make float confidence score per box 

            x1, y1, x2, y2 = boxes.xyxy[i] 

            detection = {
                "class_id": class_id, 
                "class_name": class_names[class_id], # "person" not "0"
                "confidence": confidence,
                "bounding_box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            }

            parsed_detections.append(detection)

        return parsed_detections
    

 
