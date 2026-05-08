from pathlib import Path

from vision.face_recognizer import FaceRecognizer
from vision.identity_manager import IdentityManager


recognizer = FaceRecognizer()
identity_manager = IdentityManager()


identity_name = "Steven"

image_folder = Path("backend/data/faces/Steven")


for image_path in image_folder.glob("*.jpg"):

    embedding = recognizer.generate_embedding(
        str(image_path)
    )

    identity_manager.add_identity(
        name=identity_name,
        embedding=embedding,
        authorized=True
    )

    print(f"Processed: {image_path.name}")


print(identity_manager.identities)
