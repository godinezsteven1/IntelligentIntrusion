from deepface import DeepFace

class IdentityManager:


    """
    Class IdentityManager manages identity reasoning that includes 
    authorized identities, embedding logic, and trust logic. 
    """

    def __init__(self):

     
        self.identities = {} #put known identities here 

    def add_identity(self, name, embedding, authorized=True):
    
        if name not in self.identities:
            self.identities[name] = {
                "authorized": authorized,
                "embeddings": []
            }

        self.identities[name]["embeddings"].append(embedding)
            
    def get_identity(self, name):
        return self.identities.get(name)

    def is_authorized(self,name):

        identity = self.get_identity(name)
    
        if (identity) is None:
            return False
            
        return identity["authorized"]
