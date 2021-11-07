from scripts.config import Config

def cfg():
    # Quick test  
    config = Config()
    config.set_seed() 
    
    return 0

def test_cfg():
    assert cfg() == 0
    