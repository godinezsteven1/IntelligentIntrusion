from deepface import DeepFace

class FaceRecognizer:


    """
    Class FaceRecognizer is responsible for embedding using DeepFace.
    """

    def __init__(
        self,
        model_name="Facenet" # eventually work to ArcFace
    )
        self.model_name = model_name

    def generate_embedding(self, image_path):

        embedding_result = DeepFace.represent(
            img_path=image_path,
            model_name=self.model_name,
            enforce_detection=True
        )

        embedding = embedding_result[0]["embedding"] # vector extraction

        return embedding
