class DetectionParser:

    """
    Class DetectionParser converts YOLO detections into normalized detection dictionaries.

    Output schema for each detection:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bounding_box": [x1, y1, x2, y2]
        }
    """

    def parse(self, boxes, class_names):

        parsed_detections = []

        if len(boxes.cls) == 0:
            return []

        total_detections = len(boxes.cls) #cls class labels for each box (for each labeled box)

        for i in range(total_detections): 
        
            class_id = int(boxes.cls[i]) # make integer class labels per box 
            class_name = class_names[class_id] # "person" not "0"
            confidence = float(boxes.conf[i]) # make float confidence score per box 

            x1, y1, x2, y2 = boxes.xyxy[i] 

            bounding_box = [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ]

            detection = {
                "class_id": class_id, 
                "class_name": class_name, 
                "confidence": confidence,
                "bounding_box": bounding_box
            }

            parsed_detections.append(detection)

        return parsed_detections
    

 
