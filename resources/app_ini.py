import dataclasses
import os
import re
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum


class ActivationFunction(Enum):
    SIGMOID = 'Sigmoid'
    RELU = 'ReLU'


class DistributionType(Enum):
    UNIFORM = 'Uniform'
    GAUSSIAN = 'Gaussian'


class DistributionParam(Enum):
    LB = 'lb'
    RB = 'rb'
    MEAN = 'mean'
    SD = 'sd'


@dataclass(frozen=True)
class Distribution:
    type: DistributionType
    params: dict


@dataclass(frozen=True)
class AiCfg:
    layers: tuple[int, ...]
    activation_functions: tuple[ActivationFunction, ...]
    weight_distributions: tuple[Distribution, ...]
    bias_distributions: tuple[Distribution, ...]
    learning_rate: dict[float, float | None]


@dataclass(frozen=True)
class TrainCfg:
    chunk_size: int
    chunk_count: int


@dataclass(frozen=True)
class ProcessingCfg:
    queue_max_size: int
    queue_batch_size: int


@dataclasses.dataclass(frozen=True)
class AppCfg:
    ai: AiCfg
    train: TrainCfg
    processing: ProcessingCfg


def split_n_strip(s: str):
    return [ss.strip() for ss in re.split(r',\s*(?![^()]*\))', s)]


def str_to_layers(layers: str):
    layers = split_n_strip(layers)
    layers = [int(L) for L in layers]
    return tuple(layers)


def str_to_activation_functions(funcs: str):
    func_map = {f.value: f for f in ActivationFunction}
    funcs = split_n_strip(funcs)
    funcs = [func_map[f] for f in funcs]
    return tuple(funcs)


def str_to_learning_rates(rates: str):
    rates = split_n_strip(rates)
    rates = [r.split(':') for r in rates]
    rates = {r[0]: r[1] for r in rates}
    rates = {float(k): float(v) for k, v in rates.items()}
    return rates


def str_to_distribution(distribution: str):
    ds_patterns = {
        DistributionType.UNIFORM: re.compile(fr'Uniform\s*\(\s*(?P<lb>[\d.-]+),\s*(?P<rb>[\d.-]+)\s*\)'),
        DistributionType.GAUSSIAN: re.compile(r'Gaussian\s*\(\s*(?P<mean>[\d.-]+),\s*(?P<sd>[\d.-]+)\s*\)'),
    }

    for d_type, pattern in ds_patterns.items():
        match = pattern.fullmatch(distribution)
        if match:
            props = {k: float(v) for k, v in match.groupdict().items()}
            return Distribution(d_type, props)

    raise ValueError(f'Invalid distribution: {distribution if distribution else "<empty string>"}')


def str_to_distributions(ds: str):
    ds = split_n_strip(ds)
    ds = [str_to_distribution(d) for d in ds]
    return tuple(ds)


def _get_ai_args(_cfg: ConfigParser):
    s = _cfg['AI']
    return AiCfg(
        layers=str_to_layers(s.get('layers')),
        activation_functions=str_to_activation_functions(s.get('activation functions')),
        weight_distributions=str_to_distributions(s.get('weight distributions')),
        bias_distributions=str_to_distributions(s.get('bias distributions')),
        learning_rate=str_to_learning_rates(s.get('learning rate'))
    )


def _get_train_args(_cfg: ConfigParser):
    train_section = _cfg['Train']
    return TrainCfg(
        chunk_size=train_section.getint('chunk size'),
        chunk_count=train_section.getint('chunk count')
    )


def _get_processing_args(_cfg: ConfigParser):
    processing_section = _cfg['Processing']
    return ProcessingCfg(
        queue_max_size=processing_section.getint('queue max size'),
        queue_batch_size=processing_section.getint('queue batch size')
    )


_dir = os.path.dirname(__file__)
_cfg_file = os.path.join(_dir, 'app.ini')
_cfg = ConfigParser()
_cfg.read(_cfg_file)
cfg = AppCfg(
    _get_ai_args(_cfg),
    _get_train_args(_cfg),
    _get_processing_args(_cfg)
)
