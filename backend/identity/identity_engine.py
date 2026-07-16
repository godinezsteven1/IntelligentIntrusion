

class IdentityEngine:

    """
    This class handles enrollment and recognition of known individuals 
    using InsightFace embeddings
    """ 

    

    def __init__(self):
        self.path = "backend/identity/identities"
        self.embeddings = []
