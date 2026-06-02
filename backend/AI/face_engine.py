from insightface.app import FaceAnalysis

class FaceEngine:

    
    """
    Detects faces from scene and generates embeddings 
    for persons.
    """

    def __init__(self):

        self.face_analyzer = FaceAnalysis()
        self.face_analyzer.prepare(ctx_id=0)

        
    def process(self, scene, frame): 

        faces = self.face_analyzer.get(frame)

        print(f"# Faces Detected: {len(faces)}")

        for detected_face in faces: 
            print(detected_face)
            
        for person in scene["persons"]:

            person["face"] = {
                "face_detected": False,
                "face_bbox": None,
                "embedding": None
            }
       

        return scene
        
