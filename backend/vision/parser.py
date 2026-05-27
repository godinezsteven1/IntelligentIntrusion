

def calculate_iou(box_a, box_b): #iou intersection over union which decides how associations 
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # intersection rectangle
    intersection_x1 = max(x1_a, x1_b)
    intersection_y1 = max(y1_a, y1_b)
    intersection_x2 = min(x2_a, x2_b)
    intersection_y2 = min(y2_a, y2_b)
    
    # intersection area (clamp to 0 if boxes don't overlap)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    
    # individual box areas
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    # union area (subtract intersection to avoid double-counting)
    union_area = area_a + area_b - intersection_area
    
    # guard against division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


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

        for person in persons:
            for detection in unassociated_detections:
                iou = calculate_iou(
                    person["bounding_box"],
                    detection["bounding_box"]
                )

                # overlap ratio between two bboxes [0,1]
                if iou > .1: # will fine tune later 
                    associated_object = {
                        "class_id": detection["class_id"],
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "bounding_box": detection["bounding_box"],
                        "association": "near",
                        "association_confidence": iou,
                        "iou_with_person": iou
                    } 

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "source": source,
            "persons": persons,
            "unassociated_detections": unassociated_detections
        }
    

 
