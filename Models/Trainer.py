"""Train, validate and save the best validation checkpoint."""
from pathlib import Path
import torch
from torch import nn

class Trainer:
    def __init__(self, model, optimizer, device, output_dir, logger=None, scheduler=None):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger, self.scheduler = logger, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.best_validation_accuracy = -1.0
        self.best_checkpoint = self.output_dir / "best_model.pt"
        self.global_step, self.history, self.best_epoch = 0, [], 0

    def _run_epoch(self, loader, training):
        self.model.train(training)
        total_loss = total = correct = 0
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training: self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1
                n = labels.size(0)
                total_loss += loss.item() * n
                correct += (outputs.argmax(1) == labels).sum().item()
                total += n
        return total_loss / total, correct / total

    def train(self, train_loader, val_loader, epochs, config):
        for epoch in range(1, epochs + 1):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self._run_epoch(train_loader, True)
            val_loss, val_acc = self._run_epoch(val_loader, False)
            if self.scheduler is not None: self.scheduler.step()
            row = {"epoch":epoch, "global_step":self.global_step, "train/loss":train_loss,
                   "train/accuracy":train_acc, "validation/loss":val_loss,
                   "validation/accuracy":val_acc, "learning_rate":learning_rate}
            self.history.append(row)
            if self.logger is not None: self.logger.log_epoch(row)
            if val_acc > self.best_validation_accuracy:
                self.best_validation_accuracy, self.best_epoch = val_acc, epoch
                torch.save({"model_state_dict":self.model.state_dict(), "config":config,
                            "best_validation_accuracy":val_acc, "epoch":epoch}, self.best_checkpoint)
            print(f"Epoch {epoch:03d}/{epochs} | train {train_acc*100:.2f}% / {train_loss:.4f} | val {val_acc*100:.2f}% / {val_loss:.4f}")
        return self.history

    def evaluate_test(self, test_loader):
        checkpoint = torch.load(self.best_checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        total = correct = 0
        loss_total = 0.0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss_total += self.criterion(logits, labels).item() * labels.size(0)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                true_labels.extend(labels.cpu().tolist())
                pred_labels.extend(preds.cpu().tolist())
        matrix = [[0] * 10 for _ in range(10)]
        for y, p in zip(true_labels, pred_labels): matrix[y][p] += 1
        return ({"test/loss":loss_total/total, "test/accuracy":correct/total,
                 "test/top1_accuracy_percent":100*correct/total, "test/correct":correct,
                 "test/total":total, "test/confusion_matrix":matrix}, true_labels, pred_labels)
