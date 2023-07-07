from collections import OrderedDict
import sys
import os
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple, Generator, TypedDict, Union
from typing_extensions import override

import frozenlist
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy, model_inference_strategy_dispatcher
from nemo.core.classes.common import Typing, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.classes.module import NeuralModule
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector, SaveRestoreConnector
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank, receive_generate_info, send_generate_info, synced_generate
import sentencepiece
from pytorch_lightning.plugins.environments import LightningEnvironment

from nemo.core.neural_types import NeuralType, ChannelType, LengthsType, IntType, FloatType, LogprobsType
import torch
import torch.nn.functional as F

from nemo.core.neural_types.elements import StringType, BoolType

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class InferenceModule(NeuralModule):
    def infer_input_types(self) -> Dict[str, NeuralType]:
        pass

    def infer_output_types(self) -> Dict[str, NeuralType]:
        pass


def load_tokenizer(cfg: DictConfig):
    if hasattr(cfg, "sentencepiece_legacy"):
        legacy = cfg.sentencepiece_legacy
    else:
        legacy = True if cfg.library == 'sentencepiece' else False
    
    tokenizer = get_nmt_tokenizer(
        library=cfg.library,
        model_name=cfg.type,
        tokenizer_model=cfg.model,
        vocab_file=cfg.vocab_file,
        merges_file=cfg.merge_file,
        use_fast=cfg.get('use_fast', False),
        delimiter=cfg.get('delimiter', None),
        legacy=legacy,
    )

    if cfg.get('additional_special_tokens', None) is not None:
        tokens_list = OmegaConf.to_object(cfg.additional_special_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': tokens_list})

    return tokenizer


class Gpt3TextGenerattionPreProc(InferenceModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def forward(self, prompts: List[str], tokens_to_generate: int = 50, add_BOS: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return GPTModelTextGenerationStrategy._tokenize_batch(self.tokenizer, prompts, tokens_to_generate, add_BOS)


class TextGenerattionOutputType(TypedDict):
    text: List[str]  # output text
    tokens: List[List[str]]  # output sentences borken into tokens
    logprob: List[List[float]]  # log prob of generated tokens
    full_logprob: List[List[float]]  # log prob of all the tokens in the vocab
    token_ids: List[List[int]]  # output sentence token ids
    offsets: List[List[int]]  # list of tokens start positions in text


class Gpt3TextGenerattionPostProc(InferenceModule):
    def __init__(self, tokenizer, infer_strat_post_proc_fn=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.infer_strat_post_proc_fn = infer_strat_post_proc_fn
        if self.infer_strat_post_proc_fn is None:
            self.infer_strat_post_proc_fn = lambda x : x

    def forward(
            self,
            output_ids: torch.Tensor,
    ) -> TextGenerattionOutputType:
        tokenizer = self.tokenizer
        special_tokens = set()
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
            special_tokens.add(tokenizer.pad_token)
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            special_tokens.add(tokenizer.eos_token)
        if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
            special_tokens.add(tokenizer.bos_token)
        if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
            special_tokens.add(tokenizer.cls_token)
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            special_tokens.add(tokenizer.unk_token)
        if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
            special_tokens.add(tokenizer.sep_token)
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
            special_tokens.add(tokenizer.mask_token)
        resp_sentences = []
        resp_sentences_seg = []

        output_ids = output_ids.cpu().numpy().tolist()
        for decode_token in output_ids:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            if not isinstance(tokenizer, TabularTokenizer):
                words = []
                for token in decode_token:
                    if not isinstance(token, Iterable):
                        token = [token]
                    word = tokenizer.ids_to_tokens(token)
                    if isinstance(word, Iterable):
                        word = word[0]
                    if hasattr(tokenizer.tokenizer, 'byte_decoder'):
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace'
                        )
                    words.append(word)
                resp_sentences_seg.append(words)
            else:
                words = tokenizer.text_to_tokens(sentence)
                resp_sentences_seg.append(words)

        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
                        offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)
        
        output = {}
        output['text'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        #output['logprob'] = output_logits
        #output['full_logprob'] = full_logits
        output['token_ids'] = output_ids
        output['offsets'] = all_offsets
        output = self.infer_strat_post_proc_fn(output)
        return output['text']
    

def load_gpt3_model(cfg: DictConfig):
    trainer = Trainer(plugins=[LightningEnvironment()], strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        if os.path.isdir(cfg.gpt_model_file):
            model_config = OmegaConf.load(os.path.join(cfg.gpt_model_file, 'model_config.yaml'))
        else:
            model_config = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file, trainer=trainer, return_config=True,
            )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    print('devices', cfg.trainer.devices)
    print('nodes', cfg.trainer.num_nodes)
    print('tp', cfg.tensor_model_parallel_size)
    print('pp', cfg.pipeline_model_parallel_size)

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',
        )
    else:
        raise ValueError("need at a nemo file")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model.cuda()


def generate(
    model,
    inputs=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.0,
    min_tokens_to_generate=0,
    end_strings=['<|endoftext|>'],
    **strategy_args,
):
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        strategy_args, the extra arguments are treated as inference strategy arguments
        end_strings, a list of strings to stop generation when they are encountered in the output.
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
    """
    if 'strategy' in strategy_args:
        inference_strategy = strategy_args['strategy']
    else:
        inference_strategy = model_inference_strategy_dispatcher(model, **strategy_args)
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == get_model_parallel_src_rank():
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                inputs, tokens_to_generate, add_BOS
            )

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
        )
    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
        ) = receive_generate_info()

    output = synced_generate(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        compute_attention_mask=compute_attention_mask,
        compute_logprob=compute_logprob,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        min_tokens_to_generate=min_tokens_to_generate,
        end_strings=end_strings,
    )
    output_ids, output_logits, full_logits = output
    return (output_ids, output_logits, full_logits)


# TODO NeMo doesn't support mixed parameter batches (e.g. diff top_k values for each element in batch)
def generate_text(
    model: MegatronGPTModel,
    trainer: Trainer,
    sentences: Union[Tuple[torch.Tensor, torch.Tensor], List[str]],
    tokens_to_generate: int = 2000,
    all_probs: bool = False,
    temperature: float = 0.5,
    add_BOS: bool = False,
    greedy: bool = False,
    top_k: int = 1,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_tokens_to_generate: int = 1,
    compute_logprob: bool = False,
    end_strings: List[str] = ['<extra_id_1>'],
) -> Tuple:

    if parallel_state.is_unitialized():

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()

        if model.cfg.get('transformer_engine', False):
            model.setup_transformer_engine_tp_groups()

    if isinstance(sentences, (list, tuple)):
        if isinstance(sentences[0], (str, torch.Tensor)):
            output = generate(
                model.cuda(),
                inputs=sentences,
                tokens_to_generate=tokens_to_generate,
                all_probs=all_probs,
                temperature=temperature,
                add_BOS=add_BOS,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                repetition_penalty=repetition_penalty,
                min_tokens_to_generate=min_tokens_to_generate,
                compute_logprob=compute_logprob,
                end_strings=end_strings,
            )
            return output
        elif isinstance(sentences[0], dict):
            raise NotImplementedError("json object not implemented")
        else:
            raise NotImplementedError("unknown type is not implemented")
    else:
        raise NotImplementedError("unknown type is not implemented")


class Gpt3TextGeneration(InferenceModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(
            self,
            input_ids: torch.Tensor,
            input_id_lens: torch.Tensor,
            tokens_to_generate: int = 2000,
            all_probs: bool = False,
            temperature: float = 0.5,
            add_BOS: bool = False,
            greedy: bool = False,
            top_k: int = 1,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            min_tokens_to_generate: int = 1,
            compute_logprob: bool = False,
            end_strings: List[str] = ['<|endoftext|>'],
    ):
        if greedy:
            top_k = 1
        return generate_text(
            self.model,
            self.model.trainer,
            sentences=(input_ids, input_id_lens),
            tokens_to_generate=tokens_to_generate,
            all_probs=all_probs,
            temperature=temperature,
            add_BOS=add_BOS,
            greedy=False,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_tokens_to_generate=min_tokens_to_generate,
            compute_logprob=compute_logprob,
            end_strings=end_strings,
        )


class NemoPipelineStage(Typing):
    def __init__(self, input_types: Dict[str, NeuralType] = None, output_types: Dict[str, NeuralType] = None, name: str = ""):
        self._input_types = input_types
        self._output_types = output_types
        self._name = name
        self._exec_fn = None

    @property
    def name(self):
        return self._name
    
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self._input_types
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self._output_types

    def set_exec(self, fn):
        self._exec_fn = fn

    #@typecheck
    def forward(self, **kwargs):
        if self._exec_fn is None:
            raise NotImplementedError("set_exec must be called before the forward method is called")
        return self._exec_fn(**kwargs)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class NemoPipeline(Typing):
    def __init__(self):
        self._stages: List[NemoPipeline] = []
    
    def add_stage(self, input_types: Dict[str, NeuralType] = None, output_types: Dict[str, NeuralType] = None, name: str = "", index=None):
        idx = index
        if idx is None:
            idx = len(self._stages)
        self._stages.insert(
            idx,
            NemoPipelineStage(input_types=input_types, output_types=output_types, name=name)
        )

    def register_exec(self, fn: Union[NeuralModule, Callable], stage_name=None, stage_index=None):
        if stage_index is not None:
            self._stages[stage_index].set_exec(fn)
        elif stage_name is not None:
            matches = []
            for stage in self._stages:
                if stage.name == stage_name:
                    matches.append(stage)
            if len(matches) == 1:
                stage = matches[0]
                if stage._exec_fn is not None:
                    raise Exception("exec for stage already set")
                matches[0].set_exec(fn)
            elif len(matches) > 1:
                raise Exception(f"name {stage_name} is ambiguous")
            else:
                raise Exception("stage not found")
        else:
            raise NotImplementedError("TODO implement register exec based on stage input / output types")

    def register_default_execs(self):
        pass
    
    @property
    def stages(self):
        # TODO use frozenlist?
        return self._stages

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if len(self._stages) > 0:
            return self._stages[0].input_types
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if len(self._stages) > 0:
            return self._stages[-1].output_types
        return None

    def forward(self, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, **kwargs):
        return self.forward(*kwargs)


class TextGenerationPipeline(NemoPipeline):
    def __init__(self, inference_config: DictConfig=None, load_defaults=True):
        super().__init__()
        self.inference_config = inference_config
        self.add_stage(
            input_types=OrderedDict({
                "prompts": [NeuralType(None, StringType())],
                "tokens_to_generate": NeuralType(None, IntType(), optional=True),
                "add_BOS": NeuralType(None, IntType(), optional=True)
            }),
            output_types=OrderedDict({
                "input_ids": NeuralType(('B', 'T'), ChannelType()),
                "input_id_lens": NeuralType(('B',), LengthsType()),
            }),
            name="preprocessor"
        )
        self.add_stage(
            input_types=OrderedDict({
                "input_ids": NeuralType(('B', 'T'), ChannelType()),
                "input_id_lens": NeuralType(('B',), LengthsType()),
                "tokens_to_generate": NeuralType(None, IntType(), optional=True),
                "all_probs": NeuralType(None, BoolType(), optional=True),
                "temperature": NeuralType(None, FloatType(), optional=True),
                "add_BOS": NeuralType(None, FloatType(), optional=True),
                "greedy": NeuralType(None, BoolType(), optional=True),
                "top_k": NeuralType(None, IntType(), optional=True),
                "top_p": NeuralType(None, FloatType(), optional=True),
                "repetition_penalty": NeuralType(None, FloatType(), optional=True),
                "min_tokens_to_generate": NeuralType(None, FloatType(), optional=True),
                "compute_logprob": NeuralType(None, BoolType(), optional=True),
                "end_strings": [NeuralType(None, StringType())],
            }),
            output_types=OrderedDict({
                "output_ids": NeuralType(('B', 'T'), ChannelType()),
                "logprob": NeuralType(('B', 'T'), LogprobsType(), optional=True),
                "full_logprob": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
            }),
            name="text_generation"
        )
        self.add_stage(
            input_types=OrderedDict({
                "output_ids": NeuralType(('B', 'T'), ChannelType()),
            }),
            output_types=OrderedDict({
                "text": [NeuralType(None, StringType())],
            }),
            name="postprocessor"
        )
        
        if load_defaults:
            self.register_default_execs()
    
    def register_default_execs(self):
        cfg = self.inference_config
        if os.path.isdir(cfg.gpt_model_file):
            model_config = OmegaConf.load(os.path.join(cfg.gpt_model_file, "model_config.yaml"))
        else:
            raise Exception("Should be dir")
        
        tokenizer_cfg = model_config.tokenizer
        with open_dict(tokenizer_cfg):
            if tokenizer_cfg.model is not None and tokenizer_cfg.model.startswith("nemo:"):
                tokenizer_cfg.model = os.path.join(cfg.gpt_model_file, tokenizer_cfg.model.split(":", 1)[1])
            if tokenizer_cfg.vocab_file is not None and tokenizer_cfg.vocab_file.startswith("nemo:"):
                tokenizer_cfg.vocab_file = os.path.join(cfg.gpt_model_file, tokenizer_cfg.vocab_file.split(":", 1)[1])
            if tokenizer_cfg.merge_file is not None and tokenizer_cfg.merge_file.startswith("nemo:"):
                tokenizer_cfg.merge_file = os.path.join(cfg.gpt_model_file, tokenizer_cfg.merge_file.split(":", 1)[1])

        tokenizer = load_tokenizer(tokenizer_cfg)

        model = load_gpt3_model(cfg)

        preprocessor = Gpt3TextGenerattionPreProc(tokenizer)
        text_generator = Gpt3TextGeneration(model)
        postprocessor = Gpt3TextGenerattionPostProc(tokenizer)

        self.register_exec(preprocessor, stage_name="preprocessor")
        self.register_exec(postprocessor, stage_name="postprocessor")
        self.register_exec(text_generator, stage_name="text_generation")

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return OrderedDict({
            "prompts": [NeuralType(None, StringType())],
            "tokens_to_generate": NeuralType(None, IntType(), optional=True),
            "all_probs": NeuralType(None, BoolType(), optional=True),
            "temperature": NeuralType(None, FloatType(), optional=True),
            "add_BOS": NeuralType(None, FloatType(), optional=True),
            "greedy": NeuralType(None, BoolType(), optional=True),
            "top_k": NeuralType(None, IntType(), optional=True),
            "top_p": NeuralType(None, FloatType(), optional=True),
            "repetition_penalty": NeuralType(None, FloatType(), optional=True),
            "min_tokens_to_generate": NeuralType(None, FloatType(), optional=True),
            "compute_logprob": NeuralType(None, BoolType(), optional=True),
            "end_strings": [NeuralType(None, StringType())],
        })
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return OrderedDict({
            "text": [NeuralType(None, StringType())],
            "output_ids": NeuralType(('B', 'T'), ChannelType()),
            "logprob": NeuralType(('B', 'T'), LogprobsType(), optional=True),
            "full_logprob": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
        })

    @typecheck()
    def forward(self,
        prompts=None,
        tokens_to_generate=50,
        all_probs=False,
        temperature=0.5,
        add_BOS=False,
        greedy=False,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        min_tokens_to_generate=1,
        compute_logprob=True,
        end_strings=["<|endoftext|>"]
    ):
        preprocessor = self.stages[0]
        text_generator = self.stages[1]
        postprocessor = self.stages[2]
        input_ids, lens = preprocessor(prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)
        output_ids, output_logits, full_logits = text_generator(
            input_ids=input_ids.cuda(),
            input_id_lens=lens.cuda(),
            tokens_to_generate=tokens_to_generate,
            all_probs=all_probs,
            compute_logprob=compute_logprob,
            temperature=temperature,
            greedy=greedy,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_tokens_to_generate=min_tokens_to_generate,
            end_strings=end_strings,
        )
        text = postprocessor(output_ids=output_ids)

        return text, output_ids, output_logits, full_logits


def old_main():
    gpt_model_file = "/models/gpt2b"
    infer_cfg_file = "gpt_infer.yaml"

    if os.path.isdir(gpt_model_file):
        model_config = OmegaConf.load(os.path.join(gpt_model_file, 'model_config.yaml'))
    else:
        raise Exception("Should be dir")

    tokenizer_cfg = model_config.tokenizer
    with open_dict(tokenizer_cfg):
        if tokenizer_cfg.model is not None and tokenizer_cfg.model.startswith("nemo:"):
            tokenizer_cfg.model = os.path.join(gpt_model_file, tokenizer_cfg.model.split(":", 1)[1])
        if tokenizer_cfg.vocab_file is not None and tokenizer_cfg.vocab_file.startswith("nemo:"):
            tokenizer_cfg.vocab_file = os.path.join(gpt_model_file, tokenizer_cfg.vocab_file.split(":", 1)[1])
        if tokenizer_cfg.merge_file is not None and tokenizer_cfg.merge_file.startswith("nemo:"):
            tokenizer_cfg.merge_file = os.path.join(gpt_model_file, tokenizer_cfg.merge_file.split(":", 1)[1])

    tokenizer = load_tokenizer(tokenizer_cfg)

    cfg = OmegaConf.load(infer_cfg_file)
    model = load_gpt3_model(cfg)

    preprocessor = Gpt3TextGenerattionPreProc(tokenizer)
    text_generator = Gpt3TextGeneration(model)
    postprocessor = Gpt3TextGenerattionPostProc(tokenizer)
    input_ids, lens = preprocessor(["Deep learning is", "I wish i could"])
    print(input_ids)
    print(lens)
    print("======================")

    output_ids, output_logits, full_logits = text_generator(input_ids.cuda(), lens.cuda(), tokens_to_generate=50, compute_logprob=True, all_probs=True)
    print(output_ids)
    print(output_logits)
    print("===============")

    text = postprocessor(input_ids)
    print(text)


def main():
    infer_cfg_file = "gpt_infer.yaml"
    cfg = OmegaConf.load(infer_cfg_file)

    text_generation_pipe = TextGenerationPipeline(inference_config=cfg, load_defaults=True)
    prompts = ["Deep learning is", "Is python a good programming language?"]
    text, output_ids, logits, all_logits = text_generation_pipe.forward(
        prompts=prompts,
        tokens_to_generate=cfg.inference.tokens_to_generate,
        top_k=cfg.inference.top_k,
        greedy=False,
        end_strings=["."]
    )

    print(text)

if __name__ == "__main__":
    main()