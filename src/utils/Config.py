import torch


class Config:
    """
        This is a CONFIG File for training our model on the following computer :
            - RTX 3080 10GB
            - i7-10700 CPU @ 2.90GHz
            - 32gb RAM
    """
    # Audio settings
    SAMPLE_RATE = 24000  # Hz
    MIN_AUDIO_DURATION = 10  # seconds
    MAX_AUDIO_DURATION = 20  # seconds
    
    # Text settings 
    MAX_TOKEN_LENGTH = 64

    # Model settings
    MODEL_NAME = "DiTTO-TTS"  # Replace with your model's name
    EMBEDDING_DIM = 1472  # Audio embedding dimension
    NUM_LAYERS = 1  # Number of model layers
    NHEAD = 1  # Number of attention heads

    # Training settings
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    NB_SAMPLES = 10000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BETAS = [0.9, 0.999]

    # Data settings 
    # TRAIN_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/train"
    # TEST_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/test"
    # DEV_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/dev"

    TRAIN_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/train"
    TEST_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/test"
    DEV_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/dev"
    
    @staticmethod
    def display():
        """Display the current configuration."""
        print("Audio Settings:")
        print(f"  Sample Rate: {Config.SAMPLE_RATE} Hz")
        print(f"  Min Audio Duration: {Config.MIN_AUDIO_DURATION} seconds")
        print(f"  Max Audio Duration: {Config.MAX_AUDIO_DURATION} seconds")
        print("\nText Settings:")
        print(f"  Max Token : {Config.MAX_TOKEN_LENGTH} tokens")
        print("\nModel Settings:")
        print(f"  Model Name: {Config.MODEL_NAME}")
        print(f"  Embedding Dim: {Config.EMBEDDING_DIM}")
        print(f"  Num Layers: {Config.NUM_LAYERS}")
        print(f"  Attention Heads: {Config.NHEAD}")
        print("\nTraining Settings:")
        print(f"  Batch Size: {Config.BATCH_SIZE}")
        print(f"  Learning Rate: {Config.LEARNING_RATE}")
        print(f"  Betas: {Config.BETAS}")
        print(f"  Epochs: {Config.EPOCHS}")
        print(f"  Nb samples: {Config.NB_SAMPLES}")
        print(f"  Device: {Config.DEVICE}")
        print("\nData Settings:")
        print(f"  Train path: {Config.TRAIN_PATH}")
        print(f"  Test path: {Config.TEST_PATH}")        
        print(f"  Dev path: {Config.DEV_PATH}")




