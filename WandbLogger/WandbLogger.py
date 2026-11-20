"""Compatible with uploaded WandbLogger concept; W&B handles plotting."""
import wandb
class WandbLogger:
    def __init__(self, enabled, project, config, name=None, entity=None, mode="online", group=None, job_type="train"):
        self.run = None
        if enabled and mode != "disabled":
            self.run = wandb.init(project=project, entity=entity, config=config, name=name,
                                  group=group, job_type=job_type, mode=mode, reinit="finish_previous")
            self.run.define_metric("epoch")
            for key in ("train/loss", "train/accuracy", "validation/loss", "validation/accuracy", "learning_rate"):
                self.run.define_metric(key, step_metric="epoch")
    def log_epoch(self, metrics):
        if self.run is not None: self.run.log(metrics)
    def log_summary(self, values):
        if self.run is not None:
            for key, value in values.items(): self.run.summary[key] = value
    def log_model(self, checkpoint_path):
        if self.run is not None:
            artifact = wandb.Artifact("mobilenetv2-" + self.run.id, type="model")
            artifact.add_file(str(checkpoint_path))
            self.run.log_artifact(artifact)
    def log_confusion_matrix(self, labels, predictions, class_names):
        if self.run is not None:
            self.run.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels, preds=predictions, class_names=class_names)})
    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
