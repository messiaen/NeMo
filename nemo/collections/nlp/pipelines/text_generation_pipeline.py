from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.text_generation_strategy import (
    model_inference_strategy_dispatcher,
    model_static_inference_strategy_dispatcher,
)
from nemo.collections.nlp.modules.common.text_generation_utils import generate_output_ids
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.inference_pipeline import (
    InferencePipeline,
    PipelineStage,
    PipelineStageType,
    load_inference_pipeline,
)
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, FloatType, IntType, LengthsType, LogprobsType, NeuralType
from nemo.core.neural_types.elements import BoolType, StringType

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


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


class TextGenerationStage:
    def __init__(self, model):
        self.model = model
        self.strategy = model_inference_strategy_dispatcher(self.model)

    def __call__(
        self,
        input_ids: torch.Tensor,
        input_id_lens: torch.Tensor,
        tokens_to_generate: int = 2000,
        all_probs: bool = False,
        temperature: float = 0.5,
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
        return generate_output_ids(
            self.model,
            self.strategy,
            inputs=(input_ids, input_id_lens),
            tokens_to_generate=tokens_to_generate,
            all_probs=all_probs,
            temperature=temperature,
            greedy=False,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_tokens_to_generate=min_tokens_to_generate,
            compute_logprob=compute_logprob,
            end_strings=end_strings,
        )


class TextGenerattionPreProcStage:
    def __init__(self, tokenizer, model_cls):
        self.tokenizer = tokenizer
        self._model_cls = model_cls
        self._tokenize_batch = model_static_inference_strategy_dispatcher(self._model_cls)._tokenize_batch

        self.__cur_batch = None
        self.__batch_start = {
            "prompts": [],
            "tokens_to_generate": 0,
            "add_BOS": None,
        }

    def __call__(
        self, prompts: List[str], tokens_to_generate: int = 50, add_BOS: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._tokenize_batch(self.tokenizer, prompts, tokens_to_generate, add_BOS)


class TextGenerattionPostProcStage:
    def __init__(self, tokenizer, infer_strat_post_proc_fn=None):
        self.tokenizer = tokenizer
        self.infer_strat_post_proc_fn = infer_strat_post_proc_fn
        if self.infer_strat_post_proc_fn is None:
            self.infer_strat_post_proc_fn = lambda x: x

    def __call__(
        self, output_ids: torch.Tensor,
    ):
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
        # output['logprob'] = output_logits
        # output['full_logprob'] = full_logits
        output['token_ids'] = output_ids
        output['offsets'] = all_offsets
        output = self.infer_strat_post_proc_fn(output)
        return output['text']


class TextGenerationPipeline(InferencePipeline):
    def __init__(self, inference_config: Optional[DictConfig] = None, model_config: Optional[DictConfig] = None):
        self._inference_config: DictConfig = inference_config
        self._model_config: DictConfig = model_config
        self._stages = []
        self._stages.append(
            PipelineStage(
                "preprocessor",
                input_types=OrderedDict(
                    {
                        "prompts": [NeuralType(None, StringType())],
                        "tokens_to_generate": NeuralType(None, IntType(), optional=True),
                        "add_BOS": NeuralType(None, IntType(), optional=True),
                    }
                ),
                output_types=OrderedDict(
                    {
                        "input_ids": NeuralType(('B', 'T'), ChannelType()),
                        "input_id_lens": NeuralType(('B',), LengthsType()),
                    }
                ),
                stage_type=PipelineStageType.NEMO_PROC,
            )
        )
        self._stages.append(
            PipelineStage(
                "text_generation",
                input_types=OrderedDict(
                    {
                        "input_ids": NeuralType(('B', 'T'), ChannelType()),
                        "input_id_lens": NeuralType(('B',), LengthsType()),
                        "tokens_to_generate": NeuralType(None, IntType(), optional=True),
                        "all_probs": NeuralType(None, BoolType(), optional=True),
                        "temperature": NeuralType(None, FloatType(), optional=True),
                        "greedy": NeuralType(None, BoolType(), optional=True),
                        "top_k": NeuralType(None, IntType(), optional=True),
                        "top_p": NeuralType(None, FloatType(), optional=True),
                        "repetition_penalty": NeuralType(None, FloatType(), optional=True),
                        "min_tokens_to_generate": NeuralType(None, FloatType(), optional=True),
                        "compute_logprob": NeuralType(None, BoolType(), optional=True),
                        "end_strings": [NeuralType(None, StringType())],
                    }
                ),
                output_types=OrderedDict(
                    {
                        "output_ids": NeuralType(('B', 'T'), ChannelType()),
                        "logprob": NeuralType(('B', 'T'), LogprobsType(), optional=True),
                        "full_logprob": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
                    }
                ),
                stage_type=PipelineStageType.MULTI_STEP_NNET,
            )
        )
        self._stages.append(
            PipelineStage(
                "postprocessor",
                input_types=OrderedDict({"output_ids": NeuralType(('B', 'T'), ChannelType()),}),
                output_types=OrderedDict({"text": [NeuralType(None, StringType())],}),
                stage_type=PipelineStageType.NEMO_PROC,
            )
        )

    @property
    def task_name(self) -> str:
        return "text_completion"

    @property
    def stages(self) -> List[PipelineStage]:
        return self._stages

    @property
    def inference_config(self) -> DictConfig:
        """
        Returns the inference config used to load the model / pipeline
        Same config used in load_for_inference
        """
        return self._inference_config

    @property
    def model_config(self) -> DictConfig:
        return self._model_config

    # def register_default_execs(self):
    #    tokenizer_cfg = self._model_config.tokenizer
    #    with open_dict(tokenizer_cfg):
    #        if tokenizer_cfg.model is not None and tokenizer_cfg.model.startswith("nemo:"):
    #            tokenizer_cfg.model = os.path.join(cfg.model_path, tokenizer_cfg.model.split(":", 1)[1])
    #        if tokenizer_cfg.vocab_file is not None and tokenizer_cfg.vocab_file.startswith("nemo:"):
    #            tokenizer_cfg.vocab_file = os.path.join(cfg.model_path, tokenizer_cfg.vocab_file.split(":", 1)[1])
    #        if tokenizer_cfg.merge_file is not None and tokenizer_cfg.merge_file.startswith("nemo:"):
    #            tokenizer_cfg.merge_file = os.path.join(cfg.model_path, tokenizer_cfg.merge_file.split(":", 1)[1])

    #    tokenizer = load_tokenizer(tokenizer_cfg)

    #    model = load_gpt3_model(cfg)

    #    preprocessor = Gpt3TextGenerattionPreProc(tokenizer)
    #    text_generator = Gpt3TextGeneration(model)
    #    postprocessor = Gpt3TextGenerattionPostProc(tokenizer)

    #    self.register_exec(preprocessor, stage_name="preprocessor")
    #    self.register_exec(postprocessor, stage_name="postprocessor")
    #    self.register_exec(text_generator, stage_name="text_generation")

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return OrderedDict(
            {
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
            }
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return OrderedDict(
            {
                "text": [NeuralType(None, StringType())],
                "output_ids": NeuralType(('B', 'T'), ChannelType()),
                "logprob": NeuralType(('B', 'T'), LogprobsType(), optional=True),
                "full_logprob": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
            }
        )

    @typecheck()
    def execute(
        self,
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
        end_strings=["<|endoftext|>"],
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


def main():
    infer_cfg_file = "gpt_infer.yaml"
    text_generation_pipe = load_inference_pipeline(infer_cfg_file, task_name="text_completion")
    text_generation_pipe.load_nemo_pipeline(text_generation_pipe.stage_names)

    prompts = ["Deep learning is", "Is python a good programming language?"]
    text, output_ids, logits, all_logits = text_generation_pipe.execute(
        prompts=prompts, tokens_to_generate=50, top_k=1, greedy=False, end_strings=["."]
    )

    print(text)


if __name__ == "__main__":
    main()
