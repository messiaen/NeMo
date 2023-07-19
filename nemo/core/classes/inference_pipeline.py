import importlib
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.core.classes.common import Typing, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types.neural_type import NeuralType


class PipelineStageType(Enum):
    NEMO_PROC = 1
    SINGLE_STEP_NNET = 2
    MULTI_STEP_NNET = 3


class PipelineStage(Typing):
    def __init__(
        self,
        name: str,
        input_types: Dict[str, NeuralType],
        output_types: Dict[str, NeuralType],
        stage_type: PipelineStageType = PipelineStageType.NEMO_PROC,
    ):
        self._name = name
        self._input_types = input_types
        self._output_types = output_types
        self._stage_type = (stage_type,)
        self._exec_fn = None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return self._input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return self._output_types

    @property
    def name(self):
        """Name of the stage which will be used to bind an executor using set_execute"""
        return self._name

    @property
    def type(self):
        # TODO maybe??
        return self._stage_type

    @property
    def supported_triton_backends(self):
        # TODO not sure about this
        # Some of this could be determined be export api, but not all
        return [
            "python_backend",
            "fastertransformer",
            "onnxruntime",
            # ...
        ]

    def set_execute(self, fn: Callable):
        # It would be nice if this could fail if `fn` is not compatible, but as written types are determined dyanmically
        # TODO
        # self._exec_fn = partial(typecheck(), fn, self)
        self._exec_fn = fn

    def execute(self, *args, **kwargs):
        if self._exec_fn is None:
            raise NotImplementedError()
        return self._exec_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


class InferencePipeline(Typing):
    # TODO I don't think this is needed in the interface at least
    #      keep things very simple for now
    # def __init__(
    #    self,
    #    map_location: Optional[torch.device] = None,
    #    trainer: Optional[Trainer] = None,
    # ):
    #    self._map_location = map_location
    #    self._trainer = trainer

    @property
    def task_name(self) -> str:
        raise NotImplementedError()

    @property
    def stages(self) -> List[PipelineStage]:
        raise NotImplementedError()

    @property
    def stage_names(self):
        return [stage.name for stage in self.stages]

    @property
    def inference_config(self) -> DictConfig:
        """
        Returns the inference config used to load the model / pipeline
        Same config used in load_for_inference
        """
        raise NotImplementedError()

    @property
    def model_config(self) -> DictConfig:
        raise NotImplementedError()

    # @property
    # def trainer(self):
    #    return self._trainer

    # @property
    # def map_location(self):
    #    return self._map_location

    def load_nemo_pipeline(self, parts: Optional[List[Union[str, PipelineStageType]]] = None):
        raise NotImplementedError()

    def set_stage_exec(self, stage_name: str, fn: Callable):
        for stage in self.stages:
            if stage.name == stage_name:
                stage.set_execute(fn)

    def execute(self, *args, **kwargs):
        if len(self.stages) == 0:
            raise NotImplementedError()
        output = self.stages[0].execute(*args, **kwargs)
        if len(self.stages) < 2:
            return output
        for stage in self.stages[1:]:
            output = stage.execute(*output)
        return output

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)


class InferencePipelineFactory:
    @classmethod
    def inference_pipeline(
        cls,
        task_name: Optional[str] = None,
        inference_config: Optional[DictConfig] = None,
        model_config: Optional[DictConfig] = None,
    ) -> InferencePipeline:
        raise NotImplementedError()


def load_inference_pipeline(config: Union[str, DictConfig], model_path_field_name: str = "model_path", task_name=None):
    if isinstance(config, str):
        cfg: DictConfig = OmegaConf.load(config)
    else:
        cfg = config
    if not hasattr(cfg, model_path_field_name):
        raise ValueError(f"inference config must have model path field {model_path_field_name}")
    model_path = getattr(cfg, model_path_field_name)
    model_cfg: DictConfig = ModelPT.restore_from(model_path, return_config=True)

    target_class = None
    for tgt_name in ("target", "_target_", "__target__"):
        if hasattr(model_cfg, tgt_name):
            target_class = getattr(model_cfg, tgt_name)
    if target_class is None:
        raise ValueError("Unable to find target class for model")

    module_name, class_name = target_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ImportError(f"No class {class_name} in module {module_name}")
    cls = getattr(module, class_name)
    if not issubclass(cls, InferencePipelineFactory):
        raise ValueError(f"model target class {target_class} is not a InterfacePipelineFactory")
    return cls.inference_pipeline(task_name=task_name, inference_config=cfg, model_config=model_cfg,)
