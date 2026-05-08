from vision.face_recognizer import FaceRecognizer
from vision.identity_manager import IdentityManager


recognizer = FaceRecognizer()

identity_manager = IdentityManager()


embedding = recognizer.generate_embedding(
    "backend/data/faces/Steven/steven_01.jpg"
)

identity_manager.add_identity(
    name="Steven",
    embedding=embedding,
    authorized=True
)

print(identity_manager.identities)
