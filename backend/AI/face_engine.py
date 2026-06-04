from insightface.app import FaceAnalysis

class FaceEngine:

    
    """
    Detects faces from scene and generates embeddings 
    for persons.
    """

    def __init__(self):

        self.face_analyzer = FaceAnalysis(
            name="buffalo_s",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0)

        
    def process(self, scene, frame): 

        for person in scene["persons"]:
        
            x1, y1, x2, y2 = map(int, person["bounding_box"])
        
            person_crop = frame[y1:y2, x1:x2]
        
            if person_crop.size == 0:
                continue
        
            faces = self.face_analyzer.get(person_crop)
        
            for detected_face in faces:
        
                person["face"] = {
                    "face_detected": True,
                    "embedding": detected_face.embedding
                }

        for detected_face in faces:

            face_bbox = detected_face.bbox.tolist()

            for person in scene["persons"]:

                person_bbox = person["bounding_box"]

                if self.face_inside_person(face_bbox, person_bbox):

                    person["face"] = {
                    "face_detected": True,
                    "face_bbox": face_bbox,
                    "embedding": detected_face.embedding
                    }       
      
        return scene

    def face_inside_person(self, face_bbox, person_bbox):
        
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        
        return (
            face_x1 >= person_x1 and
            face_y1 >= person_y1 and
            face_x2 <= person_x2 and
            face_y2 <= person_y2
            )
