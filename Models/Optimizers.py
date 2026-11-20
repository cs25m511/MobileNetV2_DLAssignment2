"""Optimizer factory retaining the uploaded class naming."""
import torch
class OptimizerFactory:
    @staticmethod
    def create(name, model_parameters, learning_rate, momentum=0.9, weight_decay=0.0):
        name = name.lower()
        if name == "sgd":
            return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        if name == "nesterov_sgd":
            return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        if name == "adam": return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "adamw": return torch.optim.AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "rmsprop": return torch.optim.RMSprop(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer {name}")
