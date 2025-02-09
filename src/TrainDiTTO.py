import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.DiTTO import DiTTO
from utils.Config import ConfigDiTTO, ConfigNAC
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

print(ConfigDiTTO.display())


train_set = MLSDataset(
    data_dir=ConfigDiTTO.TRAIN_PATH,
    max_text_token_length=ConfigDiTTO.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiTTO.SAMPLE_RATE,
    nb_samples=ConfigDiTTO.NB_SAMPLES,
    tokenizer_model="gpt2"
)

val_set = MLSDataset(
    data_dir=ConfigDiTTO.DEV_PATH,
    max_text_token_length=ConfigDiTTO.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiTTO.SAMPLE_RATE,
    nb_samples=ConfigDiTTO.NB_SAMPLES,
    tokenizer_model="gpt2"
)

train_loader = DataLoader(train_set,
                          batch_size=ConfigDiTTO.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=MLSDataset.collate_fn)

val_loader = DataLoader(val_set,
                        batch_size=ConfigDiTTO.BATCH_SIZE,
                        collate_fn=MLSDataset.collate_fn)



model = DiTTO(
    hidden_dim=ConfigDiTTO.HIDDEN_DIM,
    num_layers=ConfigDiTTO.NUM_LAYERS,
    num_heads=ConfigDiTTO.NUM_HEADS,
    time_dim=ConfigDiTTO.TIME_DIM,
    text_dim=ConfigDiTTO.TEXT_EMBED_DIM,
    diffusion_steps=ConfigDiTTO.DIFFUSION_STEPS,
    nac_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/NAC_epoch_20.pth"
).to(ConfigDiTTO.DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW


def train(self, train_loader):
    self.model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
        batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)
        
        text_input = batch["text"]["input_ids"]
        audio_input = batch["audio"].to(self.device)
        padding_mask_audio = batch["padding_mask_audio"].to(self.device)
        
        with torch.no_grad():
            # use NAC to get audio and text embeddings
            audio_latents, _ = self.model.nac.audio_encoder(audio_input, padding_mask_audio)
            max_length = self.model.nac.language_model.config.n_positions
            audio_latents = audio_latents[:, :, :max_length].mean(dim=1)
            text_input = text_input[:, :audio_latents.size(1)]
            text_embeddings = self.model.nac.language_model.transformer.wte(text_input)

        # Sample random time steps
        t = torch.randint(0, ConfigDiTTO.DIFFUSION_STEPS, (audio_latents.size(0),),
                          device=self.device).long()
        
        # Add noise to latents
        noise = torch.randn_like(audio_latents)
        noisy_latents = self.model.q_sample(
            audio_latents, t, noise)

        # Predict noise
        noise_pred = self.model(noisy_latents, text_embeddings, t)

        loss = self.criterion(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader), {}


def validation(self, val_loader):
    self.model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
            batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)
            
            text_input = batch["text"]["input_ids"]
            audio_input = batch["audio"].to(self.device)
            padding_mask_audio = batch["padding_mask_audio"].to(self.device)
            
            with torch.no_grad():
                # use NAC to get audio and text embeddings
                audio_latents, _ = self.model.nac.audio_encoder(audio_input, padding_mask_audio)
                max_length = self.model.nac.language_model.config.n_positions
                audio_latents = audio_latents[:, :, :max_length].mean(dim=1)
                text_input = text_input[:, :audio_latents.size(1)]
                text_embeddings = self.model.nac.language_model.transformer.wte(text_input)

            t = torch.randint(0, ConfigDiTTO.DIFFUSION_STEPS, (audio_latents.size(0),),
                              device=self.device).long()

            noise = torch.randn_like(audio_latents)
            noisy_latents = self.model.q_sample(audio_latents, t, noise)
            noise_pred = self.model(noisy_latents, text_embeddings, t)

            loss = self.criterion(noise_pred, noise)
            total_loss += loss.item()

    return total_loss / len(val_loader), {}


trainer = Trainer()
trainer.set_model(model, name=ConfigDiTTO.MODEL_NAME)\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .set_custom_functions(train_func=train, validation_func=validation)\
    .fit(
        train_data=train_loader,
        validation_data=val_loader,
        epochs=ConfigDiTTO.EPOCHS,
        learning_rate=ConfigDiTTO.LEARNING_RATE,
        checkpoint_interval=1
)
