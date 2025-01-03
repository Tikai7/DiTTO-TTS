import torch

class Config:
    # Audio settings
    SAMPLE_RATE = 24000  # Hz
    MIN_AUDIO_DURATION = 10  # seconds
    MAX_AUDIO_DURATION = 20  # seconds
    
    # Text settings 
    MAX_TOKEN = 512

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

    @staticmethod
    def display():
        """Display the current configuration."""
        print("Audio Settings:")
        print(f"  Sample Rate: {Config.SAMPLE_RATE} Hz")
        print(f"  Min Audio Duration: {Config.MIN_AUDIO_DURATION} seconds")
        print(f"  Max Audio Duration: {Config.MAX_AUDIO_DURATION} seconds")
        print("\nText Settings:")
        print(f"  Max Token : {Config.MAX_TOKEN} tokens")
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


