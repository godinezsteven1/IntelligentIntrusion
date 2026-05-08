from deepface import DeepFace

class FaceRecognizer:


    """
    Class FaceRecognizer is responsible for embedding using DeepFace.
    """

    def __init__(
        self,
        model_name="Facenet" # eventually work to ArcFace
    ):
        self.model_name = model_name

    def generate_embedding(self, image_path):

        embedding_result = DeepFace.represent(
            img_path=image_path,
            model_name=self.model_name,
            enforce_detection=True
        )

        embedding = embedding_result[0]["embedding"] # vector extraction

        return embedding

    #using cosine similarity ( range from -1 to 1 )
    def compare_embeddings(
        self,
        embedding_a,
        embedding_b
    ):
    
        embedding_a = np.array(embedding_a)
        embedding_b = np.array(embedding_b)
    
        similarity = np.dot(
            embedding_a,
            embedding_b
        ) / (
            np.linalg.norm(embedding_a)
            * np.linalg.norm(embedding_b)
        )
    
        return float(similarity)
