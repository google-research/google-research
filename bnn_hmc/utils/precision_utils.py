from jax import lax
from jax.experimental.callback import rewrite


def high_precision_dot_general(*args, **kwargs):
  kwargs.pop('precision')
  return lax.dot_general(*args, precision=lax.Precision.HIGH, **kwargs)


def high_precision_conv(*args, **kwargs):
  kwargs.pop('precision')
  kwargs.pop('lhs_shape')
  kwargs.pop('rhs_shape')
  return lax.conv_general_dilated(*args, precision=lax.Precision.HIGH, **kwargs)


HIGH_PRECISION_RULES = {
  lax.dot_general_p: high_precision_dot_general,
  lax.conv_general_dilated_p: high_precision_conv
}


def rewrite_high_precision(fn):
  return rewrite(fn, HIGH_PRECISION_RULES)