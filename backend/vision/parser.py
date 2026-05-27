
class DetectionParser:

    """
    Class DetectionParser converts YOLO detections into normalized detection dictionaries.

    Output schema for each detection:
    {
        "persons": [],
        "unassociated_detections": []
    }
    """

    def parse(
            self, 
            boxes, 
            class_names,
            frame_id,
            timestamp,
            source=0): # as of now only one source 

        persons = []
        unassociated_detections = []

        if len(boxes.cls) == 0:
            return {
                "persons": [],
                "unassociated_detections": []
            }

        total_detections = len(boxes.cls) #cls class labels for each box (for each labeled box)

        track_ids_exist = boxes.id is not None
        
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

            track_id = None
            
            if track_ids_exist:
                track_id = int(boxes.id[i])

            detection = {
                "class_id": class_id, 
                "class_name": class_name, 
                "confidence": confidence,
                "bounding_box": bounding_box
            }

            if class_name == "person": 
                detection["identity"] = {
                    "status": "unknown",
                    "name": None,
                    "embedding_distance": None,
                    "match_confidence": None,
                    "face_bbox": None
                }
                detection["person_id"] = len(persons) + 1
                detection["track_id"] = track_id
                detection["associated_objects"] = []
                persons.append(detection)
            else:
                unassociated_detections.append(detection)

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source": source,
            "persons": persons,
            "unassociated_detections": unassociated_detections
        }
    

 
