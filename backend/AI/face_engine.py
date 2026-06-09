from insightface.app import FaceAnalysis

class FaceEngine:

    
    """
    Detects faces from scene and generates embeddings 
    for persons.
    """

    def __init__(self):

        self.face_analyzer = FaceAnalysis(
            name="buffalo_s",
            #providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            providers=['CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320))  # was default (640,640)
        #self.face_analyzer.prepare(ctx_id=0)
        print("Loaded models:")
        for name, model in self.face_analyzer.models.items():
            print(name, type(model))

        
    def process(self, scene, frame): 

        for person in scene["persons"]:
        
            x1, y1, x2, y2 = map(int, person["bounding_box"])
        
            person_crop = frame[y1:y2, x1:x2]
        
            if person_crop.size == 0:
                continue
        
            #faces = self.face_analyzer.get(person_crop)
            try:
                faces = self.face_analyzer.get(person_crop)
            except Exception as e:
                print("FAILED:", e)
                continue
        
            for detected_face in faces:
        
                person["face"] = {
                    "face_detected": True,
                    "embedding": detected_face.embedding
                }

        
      
        return scene
