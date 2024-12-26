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
    NUM_LAYERS = 6  # Number of model layers
    NHEAD = 4  # Number of attention heads

    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 100

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
        print(f"  Epochs: {Config.EPOCHS}")


