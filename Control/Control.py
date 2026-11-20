"""End-to-end orchestration using the structure of uploaded Control.py."""
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from DataManager.DatasetNormalizer import DatasetNormalizer
from Models.MobileNetV2 import MobileNetV2
from Models.Optimizers import OptimizerFactory
from Models.Trainer import Trainer
from WandbLogger.WandbLogger import WandbLogger

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

class Control:
    def __init__(self): self.config_m = None
    def setconfig(self, config):
        self.config_m = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _setseed(seed):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _device(self):
        c = self.config_m
        if c.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        return torch.device("cuda" if c.device == "cuda" or (c.device == "auto" and torch.cuda.is_available()) else "cpu")

    def _data(self):
        c = self.config_m
        return DatasetNormalizer(data_dir=c.data_dir, batch_size=c.batch_size,
            num_workers=c.num_workers, seed=c.seed, download=not c.no_download,
            calc_meanstd=c.calc_meanstd, normalize_mean=c.normalize_mean,
            normalize_std=c.normalize_std, add_crop=c.add_crop, crop_padding=c.crop_padding,
            horizontal_flip_prob=c.horizontal_flip_prob, add_jitter=c.add_jitter,
            jitter_brightness=c.jitter_brightness, jitter_contrast=c.jitter_contrast,
            jitter_saturation=c.jitter_saturation, jitter_hue=c.jitter_hue,
            add_random_erasing=c.add_random_erasing, random_erasing_prob=c.random_erasing_prob,
            random_erasing_scale=c.random_erasing_scale, random_erasing_ratio=c.random_erasing_ratio)

    def _model(self):
        c = self.config_m
        return MobileNetV2(num_classes=10, width_mult=c.width_mult, dropout=c.dropout,
                           batch_norm=c.use_batchnorm).to(self._device())

    def _logger(self, config, name, job_type):
        c = self.config_m
        return WandbLogger(not c.no_wandb, c.wandb_project, config, name=name,
            entity=c.wandb_entity, mode=c.wandb_mode, group=c.wandb_group, job_type=job_type)

    def _run_directory(self):
        c = self.config_m
        return Path(c.run_dir) if c.run_dir else Path(c.output_dir) / (c.run_name or "baseline")

    def start(self):
        if self.config_m is None: raise RuntimeError("Call setconfig(config) first")
        c = self.config_m
        c.validate()
        if c.mode == "test": return self.test()
        self._setseed(c.seed)
        directory = self._run_directory()
        directory.mkdir(parents=True, exist_ok=True)
        config = c.as_dict()
        logger = self._logger(config, c.run_name or directory.name, "train")
        try:
            train_loader, val_loader, _ = self._data().build_loaders()
            self._setseed(c.seed)  # dataset loading must not alter model initialization
            model = self._model()
            optimizer = OptimizerFactory.create(c.optimizer, model.parameters(),
                c.learning_rate, c.momentum, c.weight_decay)
            if c.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
            elif c.lr_scheduler == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.step_size, gamma=c.gamma)
            else: scheduler = None
            trainer = Trainer(model, optimizer, self._device(), directory, logger, scheduler)
            history = trainer.train(train_loader, val_loader, c.epochs, config)
            with (directory / "history.csv").open("w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader(); writer.writerows(history)
            result = {"run_name":c.run_name or directory.name,
                      "best_epoch":trainer.best_epoch,
                      "best_validation_accuracy":trainer.best_validation_accuracy,
                      "checkpoint":str(trainer.best_checkpoint), "configuration":config,
                      "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
            (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
            logger.log_summary({"best_validation_accuracy":trainer.best_validation_accuracy,
                                "best_epoch":trainer.best_epoch,
                                "trainable_parameters":result["trainable_parameters"]})
            logger.log_model(trainer.best_checkpoint)
            print("Training complete:", result["run_name"], "validation accuracy:", trainer.best_validation_accuracy)
            return result
        finally: logger.finish()

    def test(self):
        c = self.config_m
        directory = self._run_directory()
        checkpoint_path = directory / "best_model.pt"
        if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        from DataManager.ArgConfig import ArgConfig
        from dataclasses import fields
        saved = checkpoint["config"]
        valid_names = {f.name for f in fields(ArgConfig)}
        saved = {k:v for k,v in saved.items() if k in valid_names}
        config = ArgConfig(**saved)
        config.data_dir = c.data_dir if c.data_dir != "./CIFAR10Dataset" else config.data_dir
        config.no_download = c.no_download
        config.device = c.device
        config.wandb_mode, config.wandb_project, config.wandb_entity = c.wandb_mode, c.wandb_project, c.wandb_entity
        config.no_wandb, config.wandb_group = c.no_wandb, c.wandb_group
        self.config_m = config
        self._setseed(config.seed)
        _, _, test_loader = self._data().build_loaders()
        self._setseed(config.seed)
        model = self._model()
        trainer = Trainer(model, None, self._device(), directory)
        logger = self._logger(config.as_dict(), directory.name + "_FINAL_TEST", "test")
        try:
            metrics, labels, predictions = trainer.evaluate_test(test_loader)
            (directory / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
            logger.log_epoch({k:v for k,v in metrics.items() if k != "test/confusion_matrix"})
            logger.log_summary({k:v for k,v in metrics.items() if k != "test/confusion_matrix"})
            logger.log_confusion_matrix(labels, predictions, CLASS_NAMES)
            print("FINAL TEST: top-1 accuracy =", f"{metrics['test/top1_accuracy_percent']:.2f}%")
            return metrics
        finally: logger.finish()
