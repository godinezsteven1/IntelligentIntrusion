import numpy as np
from pathlib import Path

class IdentityEngine:

    """
    This class handles enrollment and recognition of known individuals 
    using InsightFace embeddings
    """ 

    

    def __init__(self):
        self.path = Path("backend/identity/identities")
        self.known_identities = {}
        self.load_known_identities()

    def load_known_identities(self):
        identity_path = self.path

        if not identity_path.exists():
            print("WARNING: Identity directory not found")
            return

        for file in identity_path.iterdir():
        
            if file.suffix != ".npy":
                continue
            person_name = file.stem

            embedding = np.load(file)

            self.known_identities[person_name] = embedding
