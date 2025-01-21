import torch
import torch.nn as nn 

from model.NeuralAudioCodec import NeuralAudioCodec

from utils.Config import Config
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

from tqdm import tqdm
from torch.utils.data import DataLoader


print(Config.display())

# No need on windows
# Processing.remove_metadata_from_audio_folder(Config.TRAIN_PATH+"/"+"audio", Config.TRAIN_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.TEST_PATH+"/"+"audio", Config.TEST_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.DEV_PATH+"/"+"audio", Config.DEV_PATH+"/"+"audio_clean",)

train_set = MLSDataset(
    data_dir=Config.TRAIN_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)

val_set = MLSDataset(
    data_dir=Config.DEV_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)


test_set = MLSDataset(
    data_dir=Config.TEST_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)

train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)


model = NeuralAudioCodec(Config.LAMBDA_FACTOR)
model = model.to(Config.DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW


def train(self, train_loader):
    losses = 0
    self.model.train()

    for batch in tqdm(train_loader):
        batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
        batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)

        text = batch["text"]
        audio = batch["audio"].to(self.device)

        output = self.model(text, audio)

        reconstructed_audio = output["reconstructed_audio"]
        alignement_loss = output["alignment_loss"]

        reconstruction_loss = self.criterion(reconstructed_audio, audio)
        loss = self.compute_total_loss(reconstruction_loss, alignement_loss)

        losses += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    return losses / len(train_loader), {"accuracy" : -1}

def validation(self, validation_loader):
    losses = 0
    self.model.eval()

    for batch in tqdm(validation_loader):
        batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
        batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)
        
        text = batch["text"]
        audio = batch["audio"].to(self.device)
        output = self.model(text, audio)
        reconstructed_audio = output["reconstructed_audio"]
        alignement_loss = output["alignment_loss"]
        reconstruction_loss = self.criterion(reconstructed_audio, audio)
        loss = self.compute_total_loss(reconstruction_loss, alignement_loss)
        losses += loss.item()

    return losses / len(train_loader), {"accuracy" : -1}



trainer = Trainer()
trainer.set_model(model, name="NAC")\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .set_custom_functions(train_func=train, validation_func=validation)\
    .fit(
        train_data=train_loader, validation_data=val_loader, 
        epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE, checkpoint_interval=1        
    )