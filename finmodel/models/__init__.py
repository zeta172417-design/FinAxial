"""Final FinAxial model only; experimental actor architectures are not published."""
from .finaxial_decision_policy import FinAxialDecisionPolicy
from .finaxial_policy import FinAxialPolicyHead
from .stock_time_transformer import StockTimeTransformer, stock_vocab_sha256


def build_decision_policy(*, d_model=128, **config):
    if 'architecture_variant' in config:
        raise ValueError('this release contains only the final baseline decision architecture')
    return FinAxialDecisionPolicy(d_model=d_model, **config)


__all__ = ['FinAxialDecisionPolicy', 'FinAxialPolicyHead', 'StockTimeTransformer',
           'stock_vocab_sha256', 'build_decision_policy']
