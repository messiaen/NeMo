# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, MelLoss, PitchLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.collections.tts.torch.tts_data_types import SpeakerID
from nemo.collections.tts.mixins import (
    FastPitchAdapterModelMixin, 
    FastPitchLoraModelMixin, 
    FastPitchPrefixModelMixin,
)
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    Index,
    LengthsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils


@dataclass
class G2PConfig:
    _target_: str = "nemo.collections.tts.torch.g2ps.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.01"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-030921"
    phoneme_probability: float = 0.5


@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = G2PConfig()


@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = TextTokenizer()


class FastPitchModel(SpectrogramGenerator, Exportable, 
                     FastPitchAdapterModelMixin, FastPitchLoraModelMixin, FastPitchPrefixModelMixin):
    """FastPitch model (https://arxiv.org/abs/2006.06873) that is used to generate mel spectrogram from text."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        self.learn_alignment = cfg.get("learn_alignment", False)

        # Setup vocabulary (=tokenizer) and input_fft_kwargs (supported only with self.learn_alignment=True)
        input_fft_kwargs = {}
        if self.learn_alignment:
            self.vocab = None
            self.ds_class_name = cfg.train_ds.dataset._target_.split(".")[-1]

            if self.ds_class_name == "TTSDataset":
                self._setup_tokenizer(cfg)
                assert self.vocab is not None
                input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
                input_fft_kwargs["padding_idx"] = self.vocab.pad
            elif self.ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                logging.warning(
                    "AudioToCharWithPriorAndPitchDataset class has been deprecated. No support for"
                    " training or finetuning. Only inference is supported."
                )
                tokenizer_conf = self._get_default_text_tokenizer_conf()
                self._setup_tokenizer(tokenizer_conf)
                assert self.vocab is not None
                input_fft_kwargs["n_embed"] = len(self.vocab.tokens)
                input_fft_kwargs["padding_idx"] = self.vocab.pad
            else:
                raise ValueError(f"Unknown dataset class: {self.ds_class_name}")

        self._parser = None
        self._tb_logger = None
        self._wdb_logger = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)
        self.log_train_images = False

        loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = loss_scale
        pitch_loss_scale = loss_scale
        prosody_loss_scale = loss_scale
        kl_Loss_scale = loss_scale
        
        if "dur_loss_scale" in cfg:
            dur_loss_scale = cfg.dur_loss_scale
        if "pitch_loss_scale" in cfg:
            pitch_loss_scale = cfg.pitch_loss_scale
        if "prosody_loss_scale" in cfg:
            prosody_loss_scale = cfg.prosody_loss_scale
        if "kl_Loss_scale" in cfg:
            kl_Loss_scale = cfg.kl_Loss_scale

        self.mel_loss = MelLoss()
        self.pitch_loss = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss = DurationLoss(loss_scale=dur_loss_scale)
        self.prosody_loss = ProsodyLoss(loss_scale=prosody_loss_scale)
        self.kl_Loss_scale = kl_Loss_scale

        self.aligner = None
        if self.learn_alignment:
            self.aligner = instantiate(self._cfg.alignment_module)
            self.forward_sum_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()

        self.preprocessor = instantiate(self._cfg.preprocessor)
        input_fft = instantiate(self._cfg.input_fft, **input_fft_kwargs)
        output_fft = instantiate(self._cfg.output_fft)
        duration_predictor = instantiate(self._cfg.duration_predictor)
        pitch_predictor = instantiate(self._cfg.pitch_predictor)
        
        gst_model = instantiate(self._cfg.gst_model)
        sv_model = cfg.sv_model
        attentron_model = instantiate(self._cfg.attentron_model)
        reference_prosodyencoder = instantiate(self._cfg.reference_prosodyencoder)
        target_prosodyencoder = instantiate(self._cfg.target_prosodyencoder)
        target_prosodypredictor = instantiate(self._cfg.target_prosodypredictor)
                
        self.fastpitch = FastPitchModule(
            input_fft,
            output_fft,
            duration_predictor,
            pitch_predictor,
            self.aligner,
            
            gst_model,
            sv_model,
            attentron_model,
            reference_prosodyencoder,
            target_prosodyencoder,
            target_prosodypredictor,
            
            cfg.n_speakers,
            cfg.symbols_embedding_dim,
            cfg.pitch_embedding_kernel_size,
            cfg.n_mel_channels,
            cfg.max_token_duration,
            
            cfg.use_lookup_speaker,
            cfg.use_gst_speaker,
            cfg.use_sv_speaker,
            
            cfg.use_attentron,
            cfg.use_reference_prosodyencoder,
            cfg.use_target_prosodyencoder,
        )
        self._input_types = self._output_types = None

    def _get_default_text_tokenizer_conf(self):
        text_tokenizer: TextTokenizerConfig = TextTokenizerConfig()
        return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))

    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.vocab = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            if isinstance(self.logger, TensorBoardLogger):
                tb_logger = self.logger.experiment
            elif isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger
    
    @property
    def wdb_logger(self):
        if self._wdb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            if isinstance(self.logger, WandbLogger):
                wdb_logger = self.logger
            elif isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, WandbLogger):
                        wdb_logger = logger
                        break
            self._wdb_logger = wdb_logger
        return self._wdb_logger

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        if self.learn_alignment:
            ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]

            if ds_class_name == "TTSDataset":
                self._parser = self.vocab.encode
            elif ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                if self.vocab is None:
                    tokenizer_conf = self._get_default_text_tokenizer_conf()
                    self._setup_tokenizer(tokenizer_conf)
                self._parser = self.vocab.encode
            else:
                raise ValueError(f"Unknown dataset class: {ds_class_name}")
        else:
            self._parser = parsers.make_parser(
                labels=self._cfg.labels,
                name='en',
                unk_id=-1,
                blank_id=-1,
                do_normalize=True,
                abbreviation_version="fastpitch",
                make_table=False,
            )
        return self._parser

    def parse(self, str_input: str, normalize=True) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if normalize and self.text_normalizer_call is not None:
            str_input = self.text_normalizer_call(str_input, **self.text_normalizer_call_kwargs)

        if self.learn_alignment:
            eval_phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                eval_phon_mode = self.vocab.set_phone_prob(prob=1.0)

            # Disable mixed g2p representation if necessary
            with eval_phon_mode:
                tokens = self.parser(str_input)
        else:
            tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "ref_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "ref_spec_lens": NeuralType(('B'), LengthsType(), optional=True),
            "ref_audio": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
            "ref_audio_lens": NeuralType(('B'), LengthsType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
            "input_lens": NeuralType(('B'), LengthsType(), optional=True),
            "learn_prosody_predictor": NeuralType(optional=True),
        }
    )
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=None,
        pace=1.0,
        spec=None,
        ref_spec=None,
        ref_spec_lens=None,
        ref_audio=None,
        ref_audio_lens=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        learn_prosody_predictor=None
    ):
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=pace,
            spec=spec,
            ref_spec=ref_spec,
            ref_spec_lens=ref_spec_lens,
            ref_audio=ref_audio,
            ref_audio_lens=ref_audio_lens,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
            learn_prosody_predictor=learn_prosody_predictor
        )

    @typecheck(output_types={"spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType())})
    def generate_spectrogram(
        self, tokens: 'torch.tensor', speaker: Optional[int] = None, pace: float = 1.0, 
        ref_spec: Optional[torch.tensor] = None, ref_spec_lens: Optional[torch.tensor] = None,
        ref_audio: Optional[torch.tensor] = None, ref_audio_lens: Optional[torch.tensor] = None,
        spec: Optional[torch.tensor] = None, mel_lens: Optional[torch.tensor] = None, input_lens: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        if self.training:
            logging.warning("generate_spectrogram() is meant to be called in eval mode.")
        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        spect, *_ = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace, 
                         ref_spec=ref_spec, ref_spec_lens=ref_spec_lens,
                         ref_audio=ref_audio, ref_audio_lens=ref_audio_lens,
                         spec=spec, mel_lens=mel_lens, input_lens=input_lens)
        return spect

    def training_step(self, batch, batch_idx):
        attn_prior, durs, speaker = None, None, None
        if self.learn_alignment:
            if self.ds_class_name == "TTSDataset":
                if SpeakerID in self._train_dl.dataset.sup_data_types_set:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker, ref_audio, ref_audio_lens, ref_audio_sv, ref_audio_sv_lens = batch
                else:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _ = batch
            else:
                raise ValueError(f"Unknown vocab class: {self.vocab.__class__.__name__}")
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker, ref_audio, ref_audio_lens, ref_audio_sv, ref_audio_sv_lens = batch

        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)
        ref_mels, ref_spec_lens = self.preprocessor(input_signal=ref_audio, length=ref_audio_lens)

        mels_pred, _, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur, pitch, prosody_predict, prosody_encode, mu, logvar = self(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            ref_spec=ref_mels,
            ref_spec_lens=ref_spec_lens,
            ref_audio=ref_audio_sv,
            ref_audio_lens=ref_audio_sv_lens,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
            learn_prosody_predictor=False if self.current_epoch < (self.trainer.max_epochs // 2) else True,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss = mel_loss + dur_loss
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
            bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
            loss += ctc_loss + bin_loss

        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss += pitch_loss

        if prosody_predict is not None and prosody_encode is not None:
            prosody_loss = self.prosody_loss(prosody_predict=prosody_predict, prosody_tgt=prosody_encode)
            loss += prosody_loss 
            self.log("t_prosody_loss", prosody_loss)
            
        if mu is not None and logvar is not None:
            kl_loss = torch.mean((-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=2).sum(dim=1) / mel_lens), dim = 0) 
            kl_Loss_scale = self.kl_Loss_scale * (self.global_step + 1) / (self.trainer.estimated_stepping_batches // 4) if self.current_epoch < (self.trainer.max_epochs // 4) else self.kl_Loss_scale
            loss += kl_loss * kl_Loss_scale
            self.log("t_kl_loss_scale", kl_Loss_scale)
            self.log("t_kl_loss", kl_loss)
        
        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)
        self.log("t_dur_loss", dur_loss)
        self.log("t_pitch_loss", pitch_loss)
        if self.learn_alignment:
            self.log("t_ctc_loss", ctc_loss)
            self.log("t_bin_loss", bin_loss)

        return {"loss": loss} if batch_idx != 0 else {"loss"       : loss, 
                                                      "mel_target" : mels[0].data.cpu().float().numpy(),
                                                      "mel_predict": mels_pred[0].data.cpu().float().numpy(),
                                                      "attn_hard"  : attn_hard[0].data.cpu().float().numpy().squeeze().T if self.learn_alignment else None,
                                                      "attn_soft"  : attn_soft[0].data.cpu().float().numpy().squeeze().T if self.learn_alignment else None}
    
    def training_epoch_end(self, outputs):
        output = outputs[0]
        
        mel_target = plot_spectrogram_to_numpy(output["mel_target"])
        self.tb_logger.add_image("train_mel_target", mel_target, self.global_step,dataformats="HWC")
        self.wdb_logger.log_image(key="train_mel_target", images=[mel_target])

        mel_predict = plot_spectrogram_to_numpy(output["mel_predict"])
        self.tb_logger.add_image("train_mel_predict", mel_predict, self.global_step, dataformats="HWC")
        self.wdb_logger.log_image(key="train_mel_predict", images=[mel_predict])

        if self.learn_alignment:
            attn = plot_alignment_to_numpy(output["attn_hard"])
            self.tb_logger.add_image("train_attn", attn, self.global_step, dataformats="HWC")
            self.wdb_logger.log_image(key="train_attn", images=[attn])

            attn_soft = plot_alignment_to_numpy(output["attn_soft"])
            self.tb_logger.add_image("train_attn_soft", attn_soft, self.global_step, dataformats="HWC")
            self.wdb_logger.log_image(key="train_attn_soft", images=[attn_soft])
            
    def validation_step(self, batch, batch_idx):
        attn_prior, durs, speaker = None, None, None
                if self.learn_alignment:
            if self.ds_class_name == "TTSDataset":
                if SpeakerID in self._train_dl.dataset.sup_data_types_set:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _, speaker, ref_audio, ref_audio_lens, ref_audio_sv, ref_audio_sv_lens = batch
                else:
                    audio, audio_lens, text, text_lens, attn_prior, pitch, _ = batch
            else:
                raise ValueError(f"Unknown vocab class: {self.vocab.__class__.__name__}")
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker, ref_audio, ref_audio_lens, ref_audio_sv, ref_audio_sv_lens = batch

        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)
        ref_mels, ref_spec_lens = self.preprocessor(input_signal=ref_audio, length=ref_audio_lens)

        # Calculate val loss on ground truth durations to better align L2 loss in time
        mels_pred, _, _, log_durs_pred, pitch_pred, _, _, _, attn_hard_dur, pitch, prosody_predict, prosody_encode, mu, logvar = self(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            ref_spec=ref_mels,
            ref_spec_lens=ref_spec_lens,
            ref_audio=ref_audio_sv,
            ref_audio_lens=ref_audio_sv_lens,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=text_lens,
            learn_prosody_predictor=False if self.current_epoch < (self.trainer.max_epochs // 2) else True,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        pitch_loss = self.pitch_loss(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        loss = mel_loss + dur_loss + pitch_loss

        prosody_loss = torch.tensor([0.])
        if prosody_predict is not None and prosody_encode is not None:
            prosody_loss = self.prosody_loss(prosody_predict=prosody_predict, prosody_tgt=prosody_encode)
            loss += prosody_loss 
        
        kl_loss = torch.tensor([0.])
        if mu is not None and logvar is not None:
            kl_loss = torch.mean((-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=2).sum(dim=1) / mel_lens), dim = 0) 
            kl_Loss_scale = self.kl_Loss_scale * (self.global_step + 1) / (self.trainer.estimated_stepping_batches // 4) if self.current_epoch < (self.trainer.max_epochs // 4) else self.kl_Loss_scale
            loss += kl_loss * kl_Loss_scale
        
        return {
            "val_loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "prosody_loss": prosody_loss,
            "kl_loss": kl_loss,
            "mel_target": mels if batch_idx == 0 else None,
            "mel_pred": mels_pred if batch_idx == 0 else None,
        }

    def validation_epoch_end(self, outputs):
        
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")
        dur_loss = collect("dur_loss")
        pitch_loss = collect("pitch_loss")
        prosody_loss = collect("prosody_loss")
        kl_loss = collect("kl_loss")
        self.log("v_loss", val_loss)
        self.log("v_mel_loss", mel_loss)
        self.log("v_dur_loss", dur_loss)
        self.log("v_pitch_loss", pitch_loss)
        self.log("v_prosody_loss", prosody_loss)
        self.log("v_kl_loss", kl_loss)

        _, _, _, _, _, _, spec_target, spec_predict = outputs[0].values()

        mel_target = plot_spectrogram_to_numpy(spec_target[0].data.cpu().float().numpy())
        self.tb_logger.add_image("val_mel_target", mel_target, self.global_step, dataformats="HWC")
        self.wdb_logger.log_image(key="val_mel_target", images=[mel_target])
                          
        mel_predict = plot_spectrogram_to_numpy(spec_predict[0].data.cpu().float().numpy())
        self.tb_logger.add_image("val_mel_predict", mel_predict, self.global_step, dataformats="HWC")
        self.wdb_logger.log_image(key="val_mel_predict", images=[mel_predict])   
        
    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        if cfg.dataset._target_ == "nemo.collections.tts.torch.data.TTSDataset":
            phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                phon_mode = self.vocab.set_phone_prob(prob=None if name == "val" else self.vocab.phoneme_probability)

            with phon_mode:
                dataset = instantiate(
                    cfg.dataset,
                    text_normalizer=self.normalizer,
                    text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                    text_tokenizer=self.vocab,
                )
        else:
            dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="val")

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastpitch",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastpitch/versions/1.8.1/files/tts_en_fastpitch_align.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models

    # Methods for model exportability
    def _prepare_for_export(self, **kwargs):
        super()._prepare_for_export(**kwargs)

        # Define input_types and output_types as required by export()
        self._input_types = {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "pitch": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "pace": NeuralType(('B', 'T_text'), optional=True),
            "volume": NeuralType(('B', 'T_text')),
            "speaker": NeuralType(('B'), Index()),
        }
        self._output_types = {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T_text'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T_text'), RegressionValuesType()),
            "volume_aligned": NeuralType(('B', 'T_spec'), RegressionValuesType()),
        }

    def _export_teardown(self):
        self._input_types = self._output_types = None

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        disabled_inputs = set()
        if self.fastpitch.speaker_emb is None:
            disabled_inputs.add("speaker")
        return disabled_inputs

    @property
    def input_types(self):
        return self._input_types

    @property
    def output_types(self):
        return self._output_types

    def input_example(self, max_batch=1, max_dim=44):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.fastpitch.parameters())
        sz = (max_batch, max_dim)
        inp = torch.randint(
            0, self.fastpitch.encoder.word_emb.num_embeddings, sz, device=par.device, dtype=torch.int64
        )
        pitch = torch.randn(sz, device=par.device, dtype=torch.float32) * 0.5
        pace = torch.clamp((torch.randn(sz, device=par.device, dtype=torch.float32) + 1) * 0.1, min=0.01)
        volume = torch.clamp((torch.randn(sz, device=par.device, dtype=torch.float32) + 1) * 0.1, min=0.01)

        inputs = {'text': inp, 'pitch': pitch, 'pace': pace, 'volume': volume}

        if self.fastpitch.speaker_emb is not None:
            inputs['speaker'] = torch.randint(
                0, self.fastpitch.speaker_emb.num_embeddings, (max_batch,), device=par.device, dtype=torch.int64
            )

        return (inputs,)

    def forward_for_export(self, text, pitch, pace, volume, speaker=None):
        return self.fastpitch.infer(text=text, pitch=pitch, pace=pace, volume=volume, speaker=speaker)
