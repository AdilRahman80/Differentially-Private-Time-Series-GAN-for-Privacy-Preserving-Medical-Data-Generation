from opacus.accountants import RDPAccountant

class PrivacyAccountant:
    """
    Helper class to track privacy budget over training epochs.
    Delegates to Opacus PrivacyEngine.
    """
    def __init__(self, privacy_engine, target_delta: float):
        self.privacy_engine = privacy_engine
        self.target_delta = target_delta
        self.history = []
        
    def log_epoch(self, epoch: int):
        # account for privacy spent
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        print(f"Epoch {epoch}: Privacy spent: ε = {epsilon:.4f}, δ = {self.target_delta}")
        self.history.append({'epoch': epoch, 'epsilon': epsilon, 'delta': self.target_delta})
        return epsilon
        
    def get_history(self):
        return self.history
