import numpy as np
import copy
import torch

class FederatedDPTraining:
    """
    Simulates Federated Learning setting where multiple institutions (clients)
    collaboratively train a DP-TimeGAN.
    """
    def __init__(self, global_model, num_clients=3, fraction=1.0):
        self.global_model = global_model
        self.num_clients = num_clients
        self.fraction = fraction
        
    def create_clients(self):
        clients = []
        for _ in range(self.num_clients):
            # Create a deep copy of the model for each client
            client_model = copy.deepcopy(self.global_model)
            clients.append(client_model)
        return clients
        
    def aggregate(self, client_models):
        """
        FedAvg algorithm: Average weights from client models.
        """
        global_dict = self.global_model.generator.state_dict()
        
        for k in global_dict.keys():
            global_dict[k] = torch.stack([
                client.generator.state_dict()[k].float() for client in client_models
            ], 0).mean(0)
            
        self.global_model.generator.load_state_dict(global_dict)
        
        # In a real FL-GAN, we might average Discriminators too, or just Generators.
        # Often in FL-GANs for privacy, Discriminators stay local.
        
        return self.global_model
