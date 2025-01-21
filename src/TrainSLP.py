import torch
import torch.nn as nn 

from model.SpeechLP import SLP

from utils.Config import ConfigSLP
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

from torch.utils.data import DataLoader


print(ConfigSLP.display())

# No need on windows
# Processing.remove_metadata_from_audio_folder(ConfigSLP.TRAIN_PATH+"/"+"audio", ConfigSLP.TRAIN_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(ConfigSLP.TEST_PATH+"/"+"audio", ConfigSLP.TEST_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(ConfigSLP.DEV_PATH+"/"+"audio", ConfigSLP.DEV_PATH+"/"+"audio_clean",)

train_set = MLSDataset(
    data_dir=ConfigSLP.TRAIN_PATH,
    max_text_token_length=ConfigSLP.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigSLP.SAMPLE_RATE,
    nb_samples=ConfigSLP.NB_SAMPLES
)

val_set = MLSDataset(
    data_dir=ConfigSLP.DEV_PATH,
    max_text_token_length=ConfigSLP.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigSLP.SAMPLE_RATE,
)


test_set = MLSDataset(
    data_dir=ConfigSLP.TEST_PATH,
    max_text_token_length=ConfigSLP.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigSLP.SAMPLE_RATE,
)

train_loader = DataLoader(train_set, batch_size=ConfigSLP.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=ConfigSLP.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
test_loader = DataLoader(test_set, batch_size=ConfigSLP.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)


model = SLP(ConfigSLP.NB_CLASSES, ConfigSLP.NHEAD ,ConfigSLP.NUM_LAYERS)
model = model.to(ConfigSLP.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW

trainer = Trainer()
trainer.set_model(model, name=ConfigSLP.MODEL_NAME)\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .fit(
        train_data=train_loader, validation_data=val_loader, 
        epochs=ConfigSLP.EPOCHS, learning_rate=ConfigSLP.LEARNING_RATE, checkpoint_interval=1        
    )