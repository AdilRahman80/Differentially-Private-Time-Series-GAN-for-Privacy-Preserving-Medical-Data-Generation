from opacus.accountants.utils import get_noise_multiplier

def calculate_epsilon(epochs: int, target_delta: float, batch_size: int, sample_size: int, noise_multiplier: float) -> float:
    """
    Estimates theoretical Epsilon for a given setup without running the full PrivacyEngine.
    Useful for dashboard projection.
    """
    from opacus.accountants import RDPAccountant
    
    accountant = RDPAccountant()
    sample_rate = batch_size / sample_size
    
    # Simulate epochs
    steps = epochs * (sample_size // batch_size)
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        
    return accountant.get_epsilon(target_delta)
