import logging
import torch
from detectron2.utils.logger import log_first_n
from .text_prompt import (
    PredefinedPromptExtractor,
    LearnablePromptExtractor,
)
from .adapter import ClipAdapter, MaskFormerClipAdapter, PerPixelClipAdapter


def build_prompt_learner(cfg, task_names):
    if cfg.PROMPT_LEARNER == "predefined":
        prompt_learner = PredefinedPromptExtractor(cfg.PREDEFINED_PROMPT_TEMPLATES)
    elif cfg.PROMPT_LEARNER == "learnable":
        prompt_learner = LearnablePromptExtractor(
            prompt_dim=cfg.PROMPT_DIM,
            prompt_shape=cfg.PROMPT_SHAPE,
            task_prompt_shape=cfg.TASK_PROMPT_SHAPE,
            task_names=task_names,
        )
        if cfg.PROMPT_CHECKPOINT != "":
            checkpoint = torch.load(cfg.PROMPT_CHECKPOINT, map_location="cpu")["model"]
            missing, unexpected = prompt_learner.load_state_dict(
                {
                    ".".join(k.split(".")[2:]): v
                    for k, v in checkpoint.items()
                    if "prompt_learner" in k
                },
                strict=False,
            )
            for param in prompt_learner.parameters():
                param.requires_grad = False
            prompt_learner.with_trainable_params = False
            log_first_n(
                logging.INFO,
                "Load Prompt Learner from {}".format(cfg.PROMPT_CHECKPOINT),
                1,
            )
            log_first_n(logging.WARN, "Missing {}".format(missing), 1)
            log_first_n(logging.WARN, "Unexpected {}".format(unexpected), 1)

        else:
            trainable_params = [
                k
                for k, v in prompt_learner.named_parameters()
                if v.requires_grad == True
            ]
            log_first_n(
                logging.INFO,
                "Prompt Learner training params: {}".format(trainable_params),
                1,
            )
    else:
        raise NotImplementedError(
            "Prompt learner {} is not supported".format(cfg.PROMPT_LEARNER)
        )
    return prompt_learner
