import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datetime import datetime

class Trainer:
    """
    Trainer class to train basic model with checkpoint support
    """

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__custom_train = None
        self.__custom_validation = None
        # self.device = "cpu"

        self.history = {
            "params": {
                "lr": None,
                "epochs": None,
                "model_name": None,
            },
            "validation": {
                "loss": [],
                "accuracy": [],
            },
            "train": {
                "loss": [],
                "accuracy": [],
            },
        }

        print(f"[INFO] Model's device is : {self.device}")

    def set_criterion(self, criterion):
        self.criterion = criterion
        return self

    def set_model(self, model, name: str = "FaceViT"):
        self.model = model
        self.model.to(self.device)
        self.history["params"]["model_name"] = name
        return self

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self
    
    def set_custom_functions(self, train_func, validation_func):
        """
            Train and Validation function have to return at least (loss, metrics)
            With metrics having "accuracy" key at least
        """
        self.__custom_train = train_func
        self.__custom_validation = validation_func
        return self

    def fit(self, train_data, validation_data=None, learning_rate=1e-4, epochs=1, verbose=True, weight_decay=1e-6, checkpoint_interval=5, checkpoint_dir="checkpoints", checkpoint_path="checkpoint_epoch_1.pth"):
        assert self.model is not None, "[ERROR] set or load the model first through .set_model() or .load_model()"
        assert self.optimizer is not None, "[ERROR] set the optimizer first through .set_optimizer()"
        assert self.criterion is not None, "[ERROR] set the loss function first through .set_criterion()"
        
        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.history["params"]["lr"] = learning_rate
        self.history["params"]["epochs"] = epochs
        try:
            start_epoch = self._load_checkpoint(checkpoint_dir+"/"+checkpoint_path)
        except:
            start_epoch = 0

        best_loss = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(start_epoch, epochs):
            train_loss, train_metrics = self.__train(train_data) if self.__custom_train is None else self.__custom_train(train_data)
            val_loss, val_metrics = self.__validate(validation_data) if self.__custom_validation is None  else  self.__custom_validation(validation_data)

            self.__print_epoch(epoch, train_loss, train_metrics, val_loss, val_metrics, verbose)

            self.history["train"]["loss"].append(train_loss)
            self.history["train"]["accuracy"].append(train_metrics["accuracy"])

            self.history["validation"]["loss"].append(val_loss)
            self.history["validation"]["accuracy"].append(val_metrics["accuracy"])


            if self.history["validation"]["loss"][-1] <= best_loss:
                best_loss = self.history["validation"]["loss"][-1]
                best_model = self.model

            if (epoch + 1) % checkpoint_interval == 0:
                self.__save_checkpoint(checkpoint_dir, epoch)

        self.model = best_model
        self.__save(checkpoint_dir=checkpoint_dir)

        return self.model, self.history

    def __save_checkpoint(self, checkpoint_dir, epoch):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, checkpoint_path)
        print(f"[INFO] Checkpoint saved at {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[ERROR] Checkpoint {checkpoint_path} not found.")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        print(f"[INFO] Loaded checkpoint from {checkpoint_path} starting at epoch {start_epoch}")
        return start_epoch


    def __train(self, train_loader):
        losses = 0
        all_labels = []
        all_predictions = []
        self.model.train()

        for batch in tqdm(train_loader):
            batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
            batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)

            text = batch["text"]
            audio = batch["audio"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(text, audio)

            loss = self.criterion(output, labels)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            predictions = torch.argmax(output, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

        metrics = self.__compute_metrics(all_labels, all_predictions)
        return losses / len(train_loader), metrics


    def __validate(self, val_loader):
        losses = 0
        all_labels = []
        all_predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
                batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)

                text = batch["text"]
                audio = batch["audio"].to(self.device)
                labels = batch["label"].to(self.device)

                output = self.model(text, audio)
                loss = self.criterion(output, labels)
                losses += loss.item()

                predictions = torch.argmax(output, dim=1)
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

        metrics = self.__compute_metrics(all_labels, all_predictions)
        return losses / len(val_loader), metrics


    def __compute_metrics(self, labels, predictions):
        """
        Calculate precision, recall, and F1-score for each class.
        """
        accuracy = accuracy_score(labels, predictions)
        return {
            "accuracy": accuracy,  
        }

    def __print_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, verbose):
        """
        Print metrics for each epoch.
        """
        if verbose:
            print(
                f"[INFO] Epoch {epoch + 1}:"
                f"\n  Train -> Loss: {train_loss:.4f},"
                f" Accuracy: {train_metrics['accuracy']:.4f}"
                f"\n  Val   -> Loss: {val_loss:.4f},"
                f" Accuracy: {val_metrics['accuracy']:.4f}"
            )


    def __save(self, checkpoint_dir="checkpoints", model_path="model.pth", history_path="history.txt"):
        """
        Save the model and training history with the model name and current date.
        """

        os.makedirs(checkpoint_dir, exist_ok=True)
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_name = self.history["params"]["model_name"] if "model_name" in self.history["params"] else "model"
        
        model_path = f"{model_name}_{current_date}.pth"
        history_path = f"{model_name}_{current_date}_history.txt"
        model_path = os.path.join(checkpoint_dir, model_path)
        history_path = os.path.join(checkpoint_dir, history_path
                                    )
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.history, history_path)

        print(f"[INFO] Model and history saved at {checkpoint_dir}")
        print(f"  - Model saved as: {model_path}")
        print(f"  - History saved as: {history_path}")
