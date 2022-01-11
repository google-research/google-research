# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Framework for Supergravity models.

The definitions in this module systematize the task of analyzing the scalar
sector of various gauged extended supergravity models by providing a common
framework.

TODO(tfish): When ray-analyzing the neighborhood of a stationary point, there
seem to be major opportunities to improve efficiency of the computation.
Explore what works here. Three ideas:
  1. We should perhaps check when the deviation from
     quadratic-form stationarity to observed stationarity reaches 10%
     and proceed  more carefully from there on.
  2. We can actually start from e.g. half the typical scaling-factor
     from all the previous runs.
  3. Rather than randomly picking directions, we could use some simple ML
     to cheaply identify non-promising directions and focus more on
     potentially promising ones. (First investigations along this direction
     were not promising - but the code is still there.)
"""

# Naming deviates from PEP-8 conventions where this makes mathematics easier
# to read. Also, local variables may name-match module-global definitions,
# and we permit complex list comprehensions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name
# pylint:disable=g-complex-comprehension
#
# Also, numpy 1.20 typing might not yet be available for open-sourced
# TensorFlow.
# pylint:disable=g-import-not-at-top


import collections
import dataclasses
import math
import numbers
import re
import time

from m_theory_lib import m_util as mu

import numpy

try:
  import numpy.typing
except ImportError:
  print('This NumPy version does not yet have numpy.typing.')

import scipy.linalg
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class SUGRASignature:
  """Supergravity signature.

  Used inside `class Supergravity`, and overridden in subclasses that
  flesh out the specifics of a model.

  Attributes:
    name: The model's name.
    dim: Spacetime dimension.
    dim_scalar_manifold: Dimension of the scalar manifold,
      e.g. 70 for N=8 in D=4.
    num_model_params: Number of model-level parameters,
      such as the omega-angle for SO8c-gauging, or other such parameters
      that pick a particular Theta embedding tensor.
    scalar_masses_dV_from_right: Whether dV multiplies the Vielbein V
      from the right when computing scalar masses.
    scalar_masses_factor: Scaling factor for scalar masses.
    gravitino_masses_factor: Scaling factor for gravitino masses.
    fermion_masses_factor: Scaling factor for fermion masses.
    vector_masses_factor: Scaling factor for vector masses.
    gauge_algebra_name: Name of the gauge algebra, e.g. 'so(8)'.
    num_spurious_vector_masses: Number of mass-zero vectors
      to eliminate from spectrum.
    generator_scaling: Generator scaling. Normally 1, but may be
      different if we have to align with different conventions.
  """
  name: str
  dim: int
  dim_scalar_manifold: int
  num_model_params: int
  scalar_masses_dV_from_right: bool
  scalar_masses_factor: float
  gravitino_masses_factor: float
  fermion_masses_factor: float = numpy.nan
  vector_masses_factor: float = numpy.nan
  gauge_algebra_name: str = 'g'
  num_spurious_vector_masses: int = 0
  generator_scaling: float = 1.0


@dataclasses.dataclass(frozen=True)
class ResidualSymmetry:
  # [num_gen, P, Q], where P, Q are indices in the adjoint representation
  # of G for the G/H coset scalar manifold.
  rank: int
  all_gens: numpy.ndarray
  semisimple_subgroup_gens: numpy.ndarray
  u1_gens: numpy.ndarray


# For small numbers of dimensions,
# there is only one compact semisimple Lie algebra
# for the given dimension.
_SEMISIMPLE_BY_DIM_AND_RANK = {
    (0, 0): '',
    (3, 1): 'so(3)',
    (6, 2): 'so(3)+so(3)',
    (8, 2): 'su(3)',
    (9, 3): 'so(3)+so(3)+so(3)',
    (10, 2): 'so(5)',
    (11, 3): 'su(3)+so(3)',
    (12, 4): 'so(3)+so(3)+so(3)+so(3)',
    (13, 3): 'so(5)+so(3)',
    (14, 2): 'g_2',
    (14, 4): 'su(3)+so(3)+so(3)',
    (15, 3): 'su(4)',
    (15, 5): 'so(3)+so(3)+so(3)+so(3)+so(3)',
    # ... There are perhaps a few more cases that are un-ambiguous.
    # A further important case is (21, 3). This is unique for if it were not,
    # there would have to be some semisimple (18, 2)-algebra.
    (21, 3): r'so(7)',
    # Unfortunately, (28, 4) is not unique.
    # This could be either g2+g2 or so(8).
    }


def _summarize_spectrum(mass_eigenspaces, digits=3, num_zeros_to_remove=0):
  """Summarizes a mass spectrum."""
  fmt = f'%+.{digits}f'
  def fmt_mass(m):
    # Weird 'avoiding minus zero' hack.
    return fmt % (1e-100 + float(fmt % m))
  expanded_masses = [
      [mass] * space.shape[0] for mass, space in mass_eigenspaces]
  # pylint:disable=g-complex-comprehensions
  aggregated_masses = collections.Counter(
      fmt_mass(m) for ms in expanded_masses for m in ms)
  fmt_mass0 = fmt_mass(0)
  aggregated_masses[fmt_mass0] = (
      aggregated_masses.get(fmt_mass0, 0) - num_zeros_to_remove)
  return {m: n for m, n in aggregated_masses.items() if n > 0}


def _summarize_spectrum_text(mass_eigenspaces, digits=3,
                             num_zeros_to_remove=0):
  aggregated_masses = _summarize_spectrum(
      mass_eigenspaces, digits=digits, num_zeros_to_remove=num_zeros_to_remove)
  return ', '.join(
      f'{m}' + (f' x{count}' if count > 1 else '')
      for m, count in sorted(aggregated_masses.items(),
                             key=lambda kv: float(kv[0])))


def _summarize_spectrum_tex(mass_eigenspaces, digits=3,
                            num_zeros_to_remove=0):
  aggregated_masses = _summarize_spectrum(
      mass_eigenspaces, digits=digits, num_zeros_to_remove=num_zeros_to_remove)
  def times(n):
    return '' if n == 1 else r'_{\times %d}' % n
  return ', '.join(
      r'${%s}%s$' % (m, times(count)) for m, count in sorted(
          aggregated_masses.items(), key=lambda kv: float(kv[0])))


@dataclasses.dataclass(frozen=True)
class EquilibriumPhysics:
  """Summary of the particle physics properties of an equilibrium."""
  metadata: 'Dict[str, str]'
  potential: float
  stationarity: float
  position: numpy.ndarray
  residual_symmetry: 'Optional[ResidualSymmetry]'
  # If masses are not available for some specific particle type, this is
  # represented by the corresponding list being empty.
  num_spurious_vector_masses: int
  mass_eigenspaces_gravitinos: 'List[Tuple[float, numpy.ndarray]]'
  mass_eigenspaces_scalars: 'List[Tuple[float, numpy.ndarray]]'
  mass_eigenspaces_fermions: 'List[Tuple[float, numpy.ndarray]]'
  mass_eigenspaces_vectors: 'List[Tuple[float, numpy.ndarray]]'

  def summarize_spectra_text(self, digits=3):
    """Text-summarizes the mass-spectra."""
    txt_gravitinos = '???'
    txt_vectors = '???'
    txt_fermions = '???'
    txt_scalars = '???'
    if self.mass_eigenspaces_gravitinos:
      txt_gravitinos = (
          _summarize_spectrum_text(self.mass_eigenspaces_gravitinos,
                                   digits=digits))
    if self.mass_eigenspaces_vectors:
      txt_vectors = (
          _summarize_spectrum_text(
              self.mass_eigenspaces_vectors,
              digits=digits,
              num_zeros_to_remove=self.num_spurious_vector_masses))
    if self.mass_eigenspaces_fermions:
      txt_fermions = _summarize_spectrum_text(self.mass_eigenspaces_fermions,
                                              digits=digits)
    if self.mass_eigenspaces_scalars:
      txt_scalars = _summarize_spectrum_text(self.mass_eigenspaces_scalars,
                                             digits=digits)
    return ('\n'
            f'(m^2/m0^2)[psi]: {txt_gravitinos}\n'
            f'(m^2/m0^2)[vec]: {txt_vectors}\n'
            f'(m^2/m0^2)[chi]: {txt_fermions}\n'
            f'(m^2/m0^2)[phi]: {txt_scalars}\n')

  def summarize_spectra_tex(self, digits=3):
    """Summarizes the mass-spectra, producing latex code."""
    tex_gravitinos = '???'
    tex_vectors = '???'
    tex_fermions = '???'
    tex_scalars = '???'
    if self.mass_eigenspaces_gravitinos:
      tex_gravitinos = (
          _summarize_spectrum_tex(self.mass_eigenspaces_gravitinos,
                                  digits=digits))
    if self.mass_eigenspaces_vectors:
      tex_vectors = (
          _summarize_spectrum_tex(
              self.mass_eigenspaces_vectors,
              digits=digits,
              num_zeros_to_remove=self.num_spurious_vector_masses))
    if self.mass_eigenspaces_fermions:
      tex_fermions = _summarize_spectrum_tex(self.mass_eigenspaces_fermions,
                                             digits=digits)
    if self.mass_eigenspaces_scalars:
      tex_scalars = _summarize_spectrum_tex(self.mass_eigenspaces_scalars,
                                            digits=digits)
    return (
        r'm^2/m_0^2[\psi]:\begin{minipage}[t]{10cm}'
        r'\begin{flushleft}\scriptsize %(tex_gravitinos)s\end{flushleft}'
        r'\end{minipage}\\'
        r'm^2/m_0^2[\chi]:\begin{minipage}[t]{10cm}'
        r'\begin{flushleft}\scriptsize %(tex_fermions)s\end{flushleft}'
        r'\end{minipage}\\'
        r'm^2/m_0^2[F]:\begin{minipage}[t]{10cm}'
        r'\begin{flushleft}\scriptsize %(tex_vectors)s\end{flushleft}'
        r'\end{minipage}\\'
        r'm^2/m_0^2[\phi]:\begin{minipage}[t]{10cm}'
        r'\begin{flushleft}\scriptsize %(tex_scalars)s\end{flushleft}'
        r'\end{minipage}' % dict(
            tex_gravitinos=tex_gravitinos,
            tex_fermions=tex_fermions,
            tex_vectors=tex_vectors,
            tex_scalars=tex_scalars))


class InvalidRegionError(Exception):
  """Optimization entered an invalid region."""


class OptimizationError(Exception):
  """Optimization did not succeed."""


# TODO(tfish): Move to m_theory_lib/equilibrium_lens.py
@dataclasses.dataclass(frozen=True)
class EquilibriumLens:
  """Context for studying a specific equilibrium in detail."""
  dim: int
  n_pot_from_sampled_dir: 'Callable[numpy.typing.ArrayLike, float]'
  tf_scalars_from_sampled_dir: 'Callable[tf.Tensor, tf.Tensor]'
  tf_stat_from_sampled_dir: 'Callable[tf.Tensor, tf.Tensor]'
  tf_pot_stat: 'Callable[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]'

  def analyze_ray(self, ray, start_scaling=1.0, scaling_factor=1.6,
                  threshold_delta_potential=1e-5,
                  threshold_stationarity=1e-7,
                  timeout_sec=240,
                  gtol=1e-7,
                  report=print):
    """Analyzes stationarity along an outgoing ray.

    Args:
      ray: [dim]-array, the direction of the outgoing ray.
      start_scaling: float, the initial scaling factor.
      scaling_factor: float, the factor by which to increase scaling per step.
      threshold_delta_potential: float, by how much the scalar potential
        must differ for the solution to be considered different from the
        original.
      threshold_stationarity: float, how large a stationarity-violation is still
        considered as approximately satisfying the stationarity condition.
      timeout_sec: float, timeout in seconds for ray-analysis.
      gtol: Gradient-tolerance `gtol` parameter for minimizer.
      report: optional callable, the function to call for reporting progress,
        or `None`.
    Returns:
      The tuple (potential, stationarity, scalar_coordinates) with
      float/numpy data.
    """
    # TODO(tfish): Add functionality to optionally better discriminate between
    # 'original solution' and 'new solution', perhaps comparing (complex)
    # A1-eigenvalues.
    # TODO(tfish): Add functionality to early-stop minimization, so that we
    # do not spend a lot of effort "running off to infinity".
    if report is None:
      report = lambda _: None
    pot0 = self.n_pot_from_sampled_dir(numpy.zeros(self.dim))
    scaling = start_scaling
    t0 = time.time()
    def check_timeout():
      dt = time.time() - t0
      print('[T_Delta=%.6f T_Limit=%s]' % (dt, timeout_sec))
      if dt > timeout_sec:
        raise OptimizationError()
    while True:
      check_timeout()
      stat_opt, xs_opt = mu.tf_minimize_v2(
          self.tf_stat_from_sampled_dir,
          scaling * ray,
          default_maxiter=1000,
          default_gtol=gtol)
      pot = self.n_pot_from_sampled_dir(xs_opt)
      report('[scaling=%.3f, stat=%10.4g, pot=%.10f (pot0=%.10f)]' %
             (scaling, stat_opt, pot, pot0))
      if not abs(pot - pot0) <= threshold_delta_potential:
        report(
            '[Initial scaling=%.3f too large; halving. '
            'Ended up at: P=%.10f, S=%.6g]' %
            (scaling, pot, stat_opt))
        scaling *= .5
      else:
        break  # from initial-scaling-scan loop.
    while True:
      check_timeout()
      scaling *= scaling_factor
      stat_opt, xs_opt = mu.tf_minimize_v2(
          self.tf_stat_from_sampled_dir,
          scaling * ray, default_maxiter=200, default_gtol=gtol)
      pot = self.n_pot_from_sampled_dir(xs_opt)
      report('[scaling=%.3f, stat=%10.4g, pot=%14.10f (pot0=%.10f)]' %
             (scaling, stat_opt, pot, pot0))
      if stat_opt > 10.0:
        # Undoes both the previous re-scaling, plus a bit extra,
        # so next upscaling will go out a bit less.
        print('[walked into wild territory: stat_opt=%.6g, '
              'walking back - scaling=%.8f]' % (stat_opt, scaling))
        scaling *= scaling_factor**(-1.9)
        continue
      if stat_opt > threshold_stationarity:
        print('[handling stuck minimization: stat_opt=%.6g]' % stat_opt)
      pos_opt_wide = self.tf_scalars_from_sampled_dir(mu.tff64(xs_opt)).numpy()
      if abs(pot - pot0) >= threshold_delta_potential:
        if (len(ray) == len(pos_opt_wide) or
            stat_opt <= threshold_stationarity):
          # We did not drop any directions when trimming things down to rays, or
          # we could minimize-down-to-zero.
          return (pot, stat_opt, pos_opt_wide)
        else:
          # We did drop some directions, and on the restricted subspace,
          # could not minimize down to zero.
          print('[need to sub-descend.]')
          stat_descended, pos_descended_wide = mu.tf_minimize_v2(
              lambda t: self.tf_pot_stat(t)[-1],
              pos_opt_wide,
              default_gtol=1e-12,
              default_maxiter=200)
          pot_descended, _ = self.tf_pot_stat(mu.tff64(pos_descended_wide))
          print('[sub-descended P=%.6f, S=%.6g]' %
                (pot_descended, stat_descended))
          if (stat_descended <= threshold_stationarity and
              abs(pot_descended - pot0) >= threshold_delta_potential):
            return (pot_descended, stat_descended, pos_descended_wide)

  def _smart_explore(
      self, keras_hidden_layers, num_classes=40,
      batch_size=32,
      epochs=200,
      screening_batch_size=1000,
      tag_solution_func=(
          lambda pot_stat_pos: '%.4f/%.3f' % tuple(pot_stat_pos[:2])),
      # These must be convertible to dict().
      analyze_ray_kwargs=(('start_scaling', 20.0), ('scaling_factor', 1.25)),
      rng=None):
    """Uses ML for smarter exploration of the neighborhood of an equilibrium."""
    # Internal/explorative for now: Use some simple ML model to assess if
    # a random direction might look reasonably promising for discovering a new
    # neighbor.
    if rng is None:
      rng = numpy.random.RandomState()
    def get_random_rays(num_rays):
      rays = rng.normal(size=(num_rays, self.dim))
      l2_rays = numpy.einsum('na,na->n', rays, rays)
      # Normalize length to 1.
      return rays * l2_rays[:, numpy.newaxis]**(-.5)
    #
    model = tf.keras.models.Sequential(
        [tf.keras.Input(shape=self.dim, name='direction')] +
        keras_hidden_layers +
        [tf.keras.layers.Dense(num_classes, activation='softmax')])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    xs = list(get_random_rays(batch_size))
    list_pot_stat_pos = [
        self.analyze_ray(ray, **dict(analyze_ray_kwargs)) for ray in xs]
    tags = [tag_solution_func(psp) for psp in list_pot_stat_pos]
    id_by_tag = {}
    def registered_solution_id(tag):
      id_by_tag.setdefault(tag, len(id_by_tag))
      return id_by_tag[tag]
    ys = [registered_solution_id(tag) for tag in tags]
    while True:
      model.fit(numpy.array(xs), numpy.array(ys),
                batch_size=batch_size,
                epochs=epochs)
      screening_xs = get_random_rays(screening_batch_size)
      screening_predictions = model(screening_xs).numpy()
      screening_entropies = numpy.where(
          screening_predictions == 0, 0,
          -screening_predictions * numpy.log(screening_predictions)).sum(
              axis=-1)
      extra_xs = [
          xs for _, xs in sorted(zip(screening_entropies, screening_xs),
                                 key=lambda e_xs: e_xs[0])[:batch_size]]
      extra_pot_stat_pos = [
          self.analyze_ray(ray) for ray in extra_xs]
      extra_tags = [tag_solution_func(psp) for psp in extra_pot_stat_pos]
      extra_ys = [registered_solution_id(tag) for tag in extra_tags]
      xs.extend(extra_xs)
      ys.extend(extra_ys)
      yield id_by_tag

# Default tag-keyword-args. Exposed here since code may want to
# use expanded keyword-args that add to these.
default_tag_kwargs = (('tag', 'S'), ('digits', 7), ('scale', 1e5))


def S_id(potential, tag='S', digits=7, scale=1e5, in_brackets=None):
  """Computes solution-identifier from potential value."""
  # This is the same as S_id published alongside the
  # "SO(8) Supergravity and the Magic of Machine Learning" paper.
  #
  # This mostly "trims" the cosmological constant (so we can
  # always disambiguate by just adding more digits and keeping
  # substring identity), but nevertheless makes e.g.
  # 11.99999999999972 ==> S1200000.
  # We string-add the tag here as it may contain (TeX)-braces {}.
  if in_brackets is None:
    bracket_piece = ''
  else:
    bracket_piece = f'[{in_brackets}]'
  digits_str = f'{int(-potential * scale + 1e-4):0{digits}d}'
  return f'{tag}{bracket_piece}{digits_str}'


# The 'detailed description' LaTeX form.
_detailed_solution_template_tex = (
    r'\label{S:%(text_tag)s}'
    r'{\small\begin{longtable}{L}'
    r'{\bf %(tex_tag)s}: '
    r'%(symmetry_breaking_tex)s%(susy_long_tex)s%(bf_stable_tex)s\\'
    r'V/g^2\approx%(potential).10f,\quad %(stationarity_tex)s\\'
    r'%(spectra)s\\[6 pt]'
    r'%(position_tex)s'
    r'\end{longtable}}')

# The 'summary line in the overview table' LaTeX form.
# The final '&' cell is for manually adding "first published" references.
_summary_solution_template_tex = (
    r'{\bf %(tex_tag)s}&%(potential).10f&%(stationarity_short_tex)s&'
    r'%(symmetry_tex)s&%(susy_tex)s&%(bf_stable_yn_tex)s&\\')


def get_submanifold_embedding_fn(submanifold_embedding):
  """Returns a submanifold-params -> scalar manifold params function."""
  if submanifold_embedding is None:
    def fn_scalars_from_params(t_params):
      return t_params
  else:
    tc_scalars_from_params = mu.tff64(submanifold_embedding)
    def fn_scalars_from_params(t_params):
      return tf.einsum('na,n->a', tc_scalars_from_params, t_params)
  return fn_scalars_from_params


class SUGRA:
  """A Supergravity model.

  Class Attributes:
    signature: The SUGRASignature of this model.

  Attributes:
    tc_scalar_gens: The t_aMN generator-matrices for the scalar coset manifold,
      as a 3-index TensorFlow tensor. Noncompact generators (which are used
      to parametrize the scalar manifold) must come before compact generators.
    tc_gens: The full set of t_aMN generator-matrices,
      as a 3-index TensorFlow tensor.
    tc_gramian: 2-index TensorFlow tensor. The scalar product for the
      Lie algebra Killing form `K` restricted to the scalar manifold.
    tc_inv_gramian: 2-index TensorFlow tensor, the inverse of tc_gramian.
    tc_gramian_scale: TensorFlow float64. Scaling tc_gramian with this number
      makes the diagonal entry with maximal magnitude have magnitude 2.
    tc_gramian_onb: 2-index TensorFlow tensor. Orthonormal basis for
      the gramian `tc_gramian`.
    tc_gramian_onbi: 2-index TensorFlow tensor. The inverse of tc_gramian_onb.
  """
  # Design note: Many functions here return tuples of tf.Tensor instances.
  # This is rather low-level, and overall, a higher-level design along
  # these lines may be more appropriate:
  # - There is a class 'SupergravityBackground' whose instances have everything
  #   from EquilibriumPhysics, plus more, but is partially populated until
  #   specific entries are requested.
  # - The 'SupergravityBackground' instance refers to a SUGRA object and uses
  #   its capabilities 'to fill the gaps'.
  signature = None  # For the abstract base class. Must be overridden.

  def __init__(self, t_aMN, verbose=False,
               group_invariant_stationarity=False,
               squash_stationarity_tf_func=None,
               gauge_group_generators=None):
    """Initializes the instance.

    Args:
      t_aMN: [a, m, m]-numpy-array of G-generators for the G/H scalar coset.
        a = dim(adjoint representation), m = dim(fundamental representation).
      verbose: bool, the default verbosity-setting for calculations.
      group_invariant_stationarity: bool, whether the Killing form should
        be used to obtain a G-invariant 'length-squared of the gradient'
        stationarity-violation measure.
      squash_stationarity_tf_func: Callable[Tensor, Tensor], a R->R function
        for 'squashing' the stationarity-violation. Reasonable choices are e.g.
        tf.math.asinh, or e.g. lambda x: tf.math.asinh(x / 10).
        Using squashed({length-squared of potential-gradient}) in place of
        {length-squared of potential-gradient} often is useful for more
        effective minimum-finding as eliminating exponential growth eliminates
        very steep "hillslopes" that mislead numerical minimizers.
      gauge_group_generators: None or [a, n]-numpy-array. If provided,
        these allow automatic analysis of residual solution-symmetry.
    """
    self._verbose = verbose
    self._gauge_group_generators = gauge_group_generators
    self._group_invariant_stationarity = group_invariant_stationarity
    self._squash_stationarity_tf_func = squash_stationarity_tf_func
    dtype = (tf.float64 if isinstance(t_aMN.ravel()[0], numbers.Real)
             else tf.complex128)
    sig = self.signature
    if sig is None:
      raise ValueError(
          'Subclasses of SUGRA must override the `signature` class attribute.')
    self.tc_gens = tf.constant(t_aMN, dtype=dtype)
    self.tc_scalar_gens = self.tc_gens[:sig.dim_scalar_manifold, :, :]
    self._dim_ad, self._dim_m, _ = t_aMN.shape
    gramian = mu.nsum('aMN,bNM->ab',
                      t_aMN[:sig.dim_scalar_manifold],
                      t_aMN[:sig.dim_scalar_manifold])
    gramian_onb, gramian_onbi = mu.get_gramian_onb(gramian)
    self.tc_gramian = mu.tff64(gramian)
    self.tc_gramian_scale = mu.tff64(2 / max(abs(numpy.diag(gramian))))
    self.tc_gramian_onb = mu.tff64(gramian_onb)
    self.tc_gramian_onbi = mu.tff64(gramian_onbi)
    #
    self.tc_inv_gramian = mu.tff64(numpy.linalg.inv(gramian))

  def tf_vielbein_batched(self, t_scalars, order2=False):
    """Computes the Vielbein from batched scalar coordinates."""
    sig = self.signature
    tc_scalar_gens = self.tc_scalar_gens
    dtype = tc_scalar_gens.dtype
    t_gen = tf.einsum(
        '...v,vIJ->...JI',
        sig.generator_scaling * tf.cast(t_scalars, dtype),
        tc_scalar_gens)
    if not order2:
      return mu.tf_expm(t_gen)
    return (
        tf.reshape(
            tf.eye(self._dim_m, dtype=dtype),
            [1] * (t_gen.ndim - 2) + [self._dim_m, self._dim_m]) + t_gen +
        tf.constant(0.5, dtype=dtype) * t_gen @ t_gen)

  def tf_vielbein(self, t_scalars, order2=False):
    """Computes the Vielbein from scalar coordinates."""
    return self.tf_vielbein_batched(t_scalars[tf.newaxis, :],
                                    order2=order2)[0]

  def tf_sugra_tensors_from_vielbein_batched(self, t_V, **kwargs):
    """Computes a tuple of relevant SUGRA tensors from the batched Vielbein.

    What tensors this function returns is mostly up to the model.
    The only constraint is that the leading tuple entry is the scalar potential.
    This may e.g. include the potential, T-tensor, A1, A2, A3.

    Args:
      t_V: 2-index TensorFlow tensor, the Vielbein.
      **kwargs: Further model-specific arguments.

    Returns:
      A tuple of batched TensorFlow tensors, with the scalar potential as the
      leading entry.
    """
    raise NotImplementedError()

  def tf_sugra_tensors_from_vielbein(self, t_V, **kwargs):
    """Computes a tuple of relevant SUGRA tensors from the Vielbein.

    The default implementation wraps up a call to
    tf_sugra_tensors_from_vielbein_batched(), so subclasses that do
    not want or need to provide support for batched computations
    can just override this method and ignore the batched methods.

    Args:
      t_V: 2-index TensorFlow tensor, the Vielbein.
      **kwargs: Further model-specific arguments.
    Returns:
      A tuple of TensorFlow tensors, with the scalar potential as the
      leading entry.
    """
    batched_ret = self.tf_sugra_tensors_from_vielbein_batched(
        t_V[tf.newaxis, :, :], **kwargs)
    return tuple(tt[0] for tt in batched_ret)

  def tf_stationarity_internal(self, t_potential, t_grad_potential, t_vielbein):
    """Computes stationarity from potential, its gradient, and the Vielbein.

    This method is exposed to allow overriding in subclasses.
    It normally should not be called directly by users.

    This method must be able to handle batched potential/gradient/vielbein.
    The default implementation computes the naive length-squared of the gradient
    if the current object was created with `group_invariant_stationarity=False`,
    and otherwise the proper group-specific inner product.

    Args:
     t_potential: tf-tensor, the potential, scalar or batched scalar.
     t_grad_potential: tf-tensor, the potential-gradient, as above.
     t_vielbein: tf-tensor, the Vielbein, as above.

    Returns:
     A tf-tensor with the stationarity, batched if inputs were so.
    """
    del t_potential, t_vielbein  # Unused by the default implementation.
    if self._group_invariant_stationarity:
      t_stat = tf.einsum('...g,...h,gh->...',
                         t_grad_potential, t_grad_potential,
                         self.tc_inv_gramian / self.tc_gramian_scale)
    # This is less computational effort and mostly works just as well.
    else:
      t_stat = tf.einsum('...g,...g->...',
                         t_grad_potential, t_grad_potential)
    if self._squash_stationarity_tf_func is None:
      return t_stat
    return self._squash_stationarity_tf_func(t_stat)

  def tf_ext_sugra_tensors(self, t_scalars, with_stationarity=True, **kwargs):
    """Computes tf_sugra_tensors_from_vielbein, extended by stationarity or NaN.

    Args:
      t_scalars: [n]-tf-tensor with scalar parameters.
      with_stationarity: bool, True if `stationarity` should actually
        be computed.
      **kwargs: Forwarded to tf_sugra_tensors_from_vielbein().

    Returns:
      A tuple of tf-tensors. The last entry will be the `stationarity`,
      which will hold a NaN if `with_stationarity`==False, otherwise the
      stationarity-violation - normally, this is the naive length-squared
      of the coordinate-gradient, but subclasses may override this to provide
      some other notion of stationarity-violation.
    """
    if with_stationarity:
      tape = tf.GradientTape()
      with tape:
        tape.watch(t_scalars)
        t_vielbein = self.tf_vielbein(t_scalars)
        tensors = self.tf_sugra_tensors_from_vielbein(t_vielbein, **kwargs)
        t_pot = tensors[0]
      t_grad_pot = tape.gradient(t_pot, t_scalars)
      t_stat = self.tf_stationarity_internal(t_pot, t_grad_pot, t_vielbein)
      if self._verbose:
        print('P=%10.5f S=%.3g' % (t_pot.numpy(), t_stat.numpy()))
      return tensors + (t_stat,)
    # Otherwise, compute the other tensors and use NaN for stationarity.
    t_vielbein = self.tf_vielbein(t_scalars)
    tensors = self.tf_sugra_tensors_from_vielbein(t_vielbein, **kwargs)
    return tensors + (mu.tff64(numpy.nan),)

  def potential_and_stationarity(self, params, **kwargs):
    """Computes (float, float) potential and stationarity from positions."""
    t_pot, *_, t_stat = self.tf_ext_sugra_tensors(
        mu.tff64(params), with_stationarity=True, **kwargs)
    return t_pot.numpy(), t_stat.numpy()

  def potential(self, params, **kwargs):
    """Computes float scalar potential from positions."""
    t_pot, *_ = self.tf_ext_sugra_tensors(
        mu.tff64(params), with_stationarity=False, **kwargs)
    return t_pot.numpy()

  def stationarity(self, scalars, **kwargs):
    """Computes float stationarity-violation from positions."""
    return self.potential_and_stationarity(scalars, **kwargs)[1]

  def tf_ext_sugra_tensors_batched(self, t_scalars, with_stationarity=True,
                                   **kwargs):
    """Computes tf_sugra_tensors_from_vielbein_batched + (stationarity or NaN,).

    Args:
      t_scalars: [n]-tf-tensor with scalar parameters.
      with_stationarity: bool, True if `stationarity` should actually
        be computed.
      **kwargs: Ignored by this implementation.

    Returns:
      A tuple of tf-tensors. The last entry will be the `stationarity`,
      which will hold a NaN if `with_stationarity`==False, otherwise the
      length-squared of the gradient.
    """
    # The batched and non-batched computation here are reasonably different to
    # justify a separate implementation, rather than making non-batched wrap up
    # the batched computation.
    # In particular, subclasses may decide to not want/need batched
    # computations, in which case the tf_sugra_tensors_from_vielbein_batched()
    # method will not be overloaded.
    del kwargs  # Unused.
    # Reshape to single-batch-index.
    tb1_scalars = mu.tf_reshaped_to_1_batch_index(t_scalars, 1)
    if with_stationarity:
      tape = tf.GradientTape()
      with tape:
        tape.watch(tb1_scalars)
        t_vielbein = self.tf_vielbein_batched(tb1_scalars)
        tensors = self.tf_sugra_tensors_from_vielbein_batched(t_vielbein)
        t_pot = tensors[0][:, tf.newaxis]
      # Here, we have to remove the batch-dimension that was introduced above
      # so that we can call .batch_jacobian().
      t_grad_pot_raw = tape.batch_jacobian(t_pot, tb1_scalars)
      t_grad_pot = t_grad_pot_raw[:, 0, :]
      t_stat = self.tf_stationarity_internal(t_pot, t_grad_pot, t_vielbein)
      if self._verbose:
        for n, (pot, stat) in enumerate(zip(t_pot.numpy(), t_stat.numpy())):
          print('[B %3d] P=%10.5f S=%.3g' % (n, pot, stat))
      ret = tensors + (t_stat,)
    else:
      # Otherwise, compute the other tensors and use NaN for stationarity.
      t_vielbein = self.tf_vielbein_batched(tb1_scalars)
      tensors = self.tf_sugra_tensors_from_vielbein_batched(t_vielbein)
      ret = tensors + (tensors[0] + mu.tff64(numpy.nan),)
    # Reshape the batch-indices on the return value.
    return tuple(mu.tf_restore_batch_indices(t_x, t_scalars, 1)
                 for t_x in ret)

  def tf_gravitino_massmatrix(self, t_A1, t_potential):
    """Computes the gravitino m^2 mass matrix (unbroken SUSY has m^2=1)."""
    return (tf.einsum('ij,ik->jk', t_A1, tf.math.conj(t_A1)) *
            mu.tfc128(-self.signature.gravitino_masses_factor) /
            tf.cast(t_potential, tf.complex128))

  def tf_gravitino_masses(self, t_A1, t_potential):
    """Returns gravitino masses-squared, in descending order."""
    t_masses_squared = self.tf_gravitino_massmatrix(t_A1, t_potential)
    return tf.linalg.svd(t_masses_squared, compute_uv=False)

  def gravitino_masses(self, A1, potential):
    """Computes gravitino masses-squared from numpy A1 and potential."""
    return self.tf_gravitino_masses(
        tf.constant(A1, dtype=tf.complex128),
        mu.tfc128(potential)).numpy()

  def tf_A123(self, t_T, want_A1=True, want_A2=True, want_A3=True):
    """Computes (t_A1, t_A2, t_A3) triplet of optional TensorFlow tensors."""
    raise NotImplementedError()

  def tf_T(self, t_vielbein):
    """Computes the T-tensor from the Vielbein."""
    raise NotImplementedError()

  def tf_fermion_massmatrix(self, ts_A123, t_potential):
    # This will depend on the actual model.
    raise NotImplementedError()

  def tf_fermion_masses(self, ts_A123, t_potential):
    """Computes spin-1/2 fermion masses."""
    t_masses_squared = self.tf_fermion_massmatrix(ts_A123, t_potential)
    return tf.linalg.svd(t_masses_squared, compute_uv=False)

  def fermion_masses(self, A3, potential):
    """Computes gravitino masses-squared from numpy A3 and potential."""
    return self.tf_fermion_masses((None, None, mu.tfc128(A3)),
                                  mu.tff64(potential)).numpy()

  def tf_vector_massmatrix(self, ts_A123, t_potential):
    """Computes the vector mass matrix."""
    # This will depend on the actual model.
    raise NotImplementedError()

  def tf_scalar_massmatrix(self, t_scalars, **kwargs):
    """Computes the scalar mass-matrix."""
    t_V0 = self.tf_vielbein(t_scalars)
    sig = self.signature
    def tf_potential_dv(t_dv):
      t_dV = self.tf_vielbein(
          tf.einsum('ba,b->a', self.tc_gramian_onb, t_dv),
          order2=True)
      t_dV_x_V0 = (
          t_V0 @ t_dV if sig.scalar_masses_dV_from_right else t_dV @ t_V0)
      return self.tf_sugra_tensors_from_vielbein(t_dV_x_V0, **kwargs)[0]
    tf_hessian_pot = mu.tf_hessian(tf_potential_dv)
    hessian_scalar = tf_hessian_pot(
        mu.tff64(numpy.zeros(sig.dim_scalar_manifold))).numpy()
    pot = self.tf_sugra_tensors_from_vielbein(t_V0, **kwargs)[0].numpy()
    mm_scalar = sig.scalar_masses_factor * hessian_scalar / (- pot)
    return mm_scalar

  def generic_canonicalize_equilibrium(self, n_scalars, verbose=False,
                                       **kwargs):
    """Simplifies an equilibrium on the scalar manifold (generic algorithm).

    The generic implementation simply scans for an equivalent solution by
    running minimization for a modified objective that is of the form

    f(coords) = ((1 + A * stationarity(coords)) *
                 (B + sum(abs(coord)**P for coord in coords)) *
                 (1 + (C * (potential(coords) - potential0))**2)).

    I.e. tries to maximize the number of zero-entries in the coordinate-vector
    while trying to not leave the orbit of the current minimum.
    This works somewhat OK, but is easy to outperform with a
    tailored-to-the-problem implementation.

    The overall idea here is that while we still want a solution that
    satisfies stationarity(coords) == 0, and changing the objective function
    in general destroys such an absolute criterion, the particular sort of
    change we introduce here is engineered to ensure that, with suitable
    parameters A, B, C, P (that do not require much fine-tuning), we only
    lose some numerical accuracy yet still maintain stationarity(coords) == 0.

    The basic mathematical idea is that if we have a f(xs) = 0 = f'(xs)
    minimum-at-zero, then the function xs -> (1 + f(xs)) * (1 + g(xs))
    will have derivative xs -> f'(xs) * (1 + g(xs)) + (1 + f(xs)) * g'(xs),
    which is in particular zero where f'(xs) == 0 == g'(xs).

    Args:
      n_scalars: numpy-vector, coordinates of the solution to be canonicalized.
      verbose: Whether to verbosely print progress information.
      **kwargs: Canonicalization parameters. This part of the API is not
        fully fixed by the base class, different subclasses may use very
        different approaches here. The base class implementation uses these
        arguments, but subclasses are free to use entirely different parameters:
        - `coeffs_from_scalars`: An optional [a, n]-numpy-matrix that maps the
          n input-parameters in `n_scalars` to a length-a coordinate-vector for
          the scalar manifold. This is useful for canonicalization on some
          submanifold with residual symmetry.
        - `scaling_stationarity`: The A-parameter in the formula above.
          Defaults to 10.0.
        - `offset_coords`: The B-parameter in the formula above.
          Should be somewhat larger than the typical coordinate-dependent
          term it is used as an offset on. If this is not provided, a value
          that should work well for first-step refinement is picked
          heuristically.
        - `coord_power`: The P-parameter in the formula above.
          L1-regularization-for-sparsification would suggest P=1,
          but any value 1 <= P < 2 will work to some extent,
          and 1 < P is nicer to handle for the 2nd-order-method optimizer.
          Defaults to 1.5.
        - `scaling_delta_potential`: The C-parameter in the formula above.
          This punishes moving away from the orbit of the current solution
          under minimization and typically is large, as we want to generally
          punish even deviations in the potential as large as 1e-5.
          Defaults to 1e4.
        - `tf_minimize_kwargs`: Keyword arguments to pass on to
          m_util.tf_minimize().
        - `tolerance`: Maximal permitted stationarity-violation or change in
          potential-value for the canonicalization-result to be considered ok.
        - `sugra_kwargs`: Keyword arguments to pass on to methods such as
          self.potential(). Defaults to {}.

    Returns:
      `None` if canonicalization failed, a numpy-vector of canonicalized
      coordinates otherwise.
    """
    num_scalars = self.signature.dim_scalar_manifold
    coeffs_from_scalars = kwargs.get('coeffs_from_scalars')
    if coeffs_from_scalars is None:
      coeffs_from_scalars = numpy.eye(num_scalars)
    # We generally would use 1.0 for "sparsity-encouraging L1-regularization",
    # but this does not play along too nicely with a 2nd order optimizer
    # like BFGS. So, a power of 1.5 is observed to generally do a better job
    # here for almost-solved problems.
    coord_power = kwargs.get('coord_power', 1.5)
    # The scalar potential is usually good to 6+ digits, so scaling up
    # the delta of the vacuum energy density by at least this much is
    # appropriate to also enforce it to not change - which would amount to
    # accidentally crossing over to another 'nearby' solution.
    scaling_delta_potential = kwargs.get('scaling_delta_potential', 1e4)
    sugra_kwargs = kwargs.get('sugra_kwargs', {})
    tc_coeffs_from_scalars = tf.reshape(mu.tff64(coeffs_from_scalars),
                                        [-1, num_scalars])
    tc_coord_power = mu.tff64(coord_power)
    @tf.function
    def tf_get_pos(t_scalars):
      return tf.einsum('ck,k->c', tc_coeffs_from_scalars, t_scalars)
    def get_pos(n_scalars):
      return numpy.einsum('ck,k->c', coeffs_from_scalars, n_scalars)
    @tf.function
    def tf_coord_score(t_pos):
      return tf.reduce_sum(
          tf.math.pow(tf.math.abs(tf_get_pos(t_pos)),
                      tc_coord_power))
    #
    pot0, stat0 = self.potential_and_stationarity(get_pos(n_scalars),
                                                  **sugra_kwargs)
    offset_coords = kwargs.get('offset_coords')
    if offset_coords is None:
      offset_coords = 1.5 * tf_coord_score(mu.tff64(n_scalars)).numpy()
    tc_offset_coords = mu.tff64(offset_coords)
    if verbose:
      print('[canon] pot=%.8f, stat=%.6g offset_coords=%.6f' % (
          pot0, stat0, offset_coords))
    tc_stat_scaling = mu.tff64(kwargs.get('scaling_stationarity', 10.0))
    tc_scaling_delta_potential = mu.tff64(scaling_delta_potential)
    tc_pot0, *_ = self.tf_ext_sugra_tensors(mu.tff64(n_scalars), **sugra_kwargs)
    def tf_loss(t_scalars):
      t_pot, *_, t_stat = self.tf_ext_sugra_tensors(t_scalars, **sugra_kwargs)
      t_delta_pot = (t_pot - tc_pot0) * tc_scaling_delta_potential
      t_factor_stat = (1 + tc_stat_scaling * t_stat)
      t_factor_coords = (tc_offset_coords + tf.reduce_sum(
          tf.math.pow(tf.math.abs(tf_get_pos(t_scalars)), tc_coord_power)))
      t_factor_potential = mu.tff64(1.0) + tf.math.square(t_delta_pot)
      t_ret = t_factor_stat * t_factor_coords * t_factor_potential
      if verbose:
        print('[canon]: %.8f [%16.12f, %16.12f, %16.12f]' % (
            t_ret.numpy(), t_factor_stat.numpy(), t_factor_coords.numpy(),
            t_factor_potential.numpy()))
      return t_ret
    tf_minimize_kwargs = dict(default_gtol=1e-15, default_maxiter=10**5)
    tf_minimize_kwargs.update(kwargs.get('tf_minimize_kwargs', {}))
    _, opt_xs = mu.tf_minimize_v2(tf_loss, n_scalars, **tf_minimize_kwargs)
    pot1, stat1 = self.potential_and_stationarity(get_pos(opt_xs),
                                                  **sugra_kwargs)
    if verbose:
      print('[canon]: Post-canonicalization P=%.8f, S=%.6g\nPosition: %r' % (
          pot1, stat1, opt_xs.tolist()))
    tolerance = kwargs.get('tolerance', 1e-6)
    if (stat1 >= tolerance or abs(pot1 - pot0) >= tolerance):
      return None  # Canonicalization failed.
    return opt_xs

  def canonicalize_equilibrium(self, n_scalars, **kwargs):
    """Simplifies an equilibrium on the scalar manifold.

    Depending on the model, there are specific techniques available to
    effectively find a solution on the gauge group orbit that is
    equivalent to the given one - and hence, subclass
    re-implementations of this method may use different keyword arguments
    to fine-tune behavior. Often, such 'canonicalization' involves finding
    a rotation that diagonalizes a symmetric matrix, and then also applying
    the same rotation to a different representation via
    decomposing-and-rebuilding the rotation for a different linear
    representation.

    Args:
      n_scalars: numpy-vector, coordinates of the solution to be canonicalized.
      **kwargs: Canonicalization parameters. This part of the API is not
        fully fixed by the base class, different subclasses may use very
        different approaches here. The base class implementation just calls the
        .generic_canonicalize_equilibrium() method. The meaning of this
        parameter for this implementation is documented there.

    Returns:
      `None` if canonicalization failed, otherwise a canonicalized
      position-vector.
    """
    return self.generic_canonicalize_equilibrium(n_scalars, **kwargs)

  def find_equilibrium(
      self, n_params,
      verbosity='',
      minimize_kwargs=None,
      # TODO(tfish):
      # Index-order should be the other way round for submanifold_embedding,
      # should be [70, n] for D=4.
      submanifold_embedding=None,
      exit_condition=None,
      report=print,
      **kwargs):
    """Finds an equilibrium via minimization."""
    fn_scalars_from_params = get_submanifold_embedding_fn(
        submanifold_embedding)
    #
    def tf_stat_func(t_params):
      t_scalars = fn_scalars_from_params(t_params)
      tensors = self.tf_ext_sugra_tensors(t_scalars, **kwargs)
      t_stationarity = tensors[-1]
      if 'S' in verbosity:
        report('P=%.10f S=%.10g' % (tensors[0].numpy(), t_stationarity.numpy()))
      if exit_condition is not None and exit_condition(tensors):
        raise InvalidRegionError()
      return t_stationarity
    try:
      _, opt_params = mu.tf_minimize_v2(tf_stat_func, n_params,
                                        **(minimize_kwargs or {}))
    except InvalidRegionError:
      return 0.0, numpy.inf, n_params
    ext_sugra_tensors = self.tf_ext_sugra_tensors(
        fn_scalars_from_params(mu.tff64(opt_params)),
        **kwargs)
    pot = ext_sugra_tensors[0].numpy()
    stat = ext_sugra_tensors[-1].numpy()
    return pot, stat, opt_params

  def get_generator_x0s(self, seed=0, scale=0.1, dim=None):
    """Yields x0-positions to start minimization from.

    Args:
      seed: The numpy.random.RandomState() seed to use.
      scale: Standard deviation for the normal distributed coordinate entries.
      dim: Optional, number of parameters.
        Defaults to the self.sig.dim_scalar_manifold.
    """
    sig = self.signature
    dim = dim or sig.dim_scalar_manifold
    rng = numpy.random.RandomState(seed=seed)
    while True:
      yield rng.normal(size=(dim,),
                       scale=scale)

  def scan(self,
           x0s=None,
           minimize_kwargs=None,
           submanifold_embedding=None,
           stat_threshold=0.01,
           verbosity='',  # 'S': Step, 'F': Final
           **kwargs):
    """Scans for equilibria."""
    submanifold_embedding = (None if submanifold_embedding is None
                             else numpy.asarray(submanifold_embedding))
    dim = (None if submanifold_embedding is None
           else submanifold_embedding.shape[0])
    if x0s is None:
      x0s = self.get_generator_x0s(dim=dim)
    for x0 in x0s:
      try:
        ret = self.find_equilibrium(
            x0, verbosity=verbosity,
            submanifold_embedding=submanifold_embedding,
            minimize_kwargs=minimize_kwargs, **kwargs)
        pot, stat, _ = ret
        if stat > stat_threshold:
          if 'F' in verbosity:
            print(f'Skipping: stat_min={stat}, pot={pot}')
          continue
        yield ret
      except Exception:  # pylint:disable=broad-except
        # Various things can have gone wrong here, such as the optimizer
        # encountering NaN in the objective function from an intermediate
        # overflow and subsequent inf/inf. We just ignore this one trial
        # and continue with the next.
        pass

  # TODO(tfish): Refactor, using equilibrium_lens.py.
  def get_equilibrium_lens(self,
                           n_equilibrium,
                           hessian_eigval_threshold=1e-3,
                           constrain_to_nondegenerate_hessian=True,
                           hessian_eigval_zero_substitute=None,
                           **kwargs):
    """Returns an EquilibriumLens for studying a particular equilibrium."""
    def tf_stat(t_pos):
      return self.tf_ext_sugra_tensors(t_pos, **kwargs)[-1]
    def tf_pot_stat(t_pos):
      t_pot, *_, t_stat = self.tf_ext_sugra_tensors(t_pos, **kwargs)
      return t_pot, t_stat
    tf_grad_stat = mu.tf_grad(tf_stat)
    tf_hessian_stat = mu.tf_jacobian(tf_grad_stat)
    tc_eq = mu.tff64(n_equilibrium)
    n_hessian_equilibrium = tf_hessian_stat(tc_eq).numpy()
    h_eigvals, h_eigvecsT = scipy.linalg.eigh(n_hessian_equilibrium)
    # Let's apply a linear transform that brings the Hessian to
    # +1/0/-1 Sylvester Normal Form.
    is_zero_eigval = abs(h_eigvals) <= hessian_eigval_threshold
    num_zero_eigvals = is_zero_eigval.sum()
    the_zero_substitute = hessian_eigval_zero_substitute
    if hessian_eigval_zero_substitute is None:
      the_zero_substitute = numpy.mean(
          [ev for ev in h_eigvals if abs(ev) > hessian_eigval_threshold])
    scaling_factors = numpy.array([
        (the_zero_substitute if z else ev)**(-.5)
        for ev, z in zip(h_eigvals, is_zero_eigval)])
    scaled_eigvecsT = numpy.einsum('n,an->an', scaling_factors, h_eigvecsT)
    sampled_eigvecs_offset = (
        num_zero_eigvals if constrain_to_nondegenerate_hessian else 0)
    n_scalar_dir_from_sampled_dir = scaled_eigvecsT[:, sampled_eigvecs_offset:]
    tc_scalar_dir_from_sampled_dir = mu.tff64(n_scalar_dir_from_sampled_dir)
    #
    def tf_scalars_from_sampled_dir(t_sdir):
      return tc_eq + tf.einsum('aS,S->a',
                               tc_scalar_dir_from_sampled_dir, t_sdir)
    def tf_stat_from_sampled_dir(t_sdir):
      return tf_stat(tf_scalars_from_sampled_dir(t_sdir))
    def n_pot_from_sampled_dir(n_sdir):
      t_pos = tf_scalars_from_sampled_dir(mu.tff64(n_sdir))
      return self.tf_ext_sugra_tensors(
          t_pos,
          with_stationarity=False,
          **kwargs)[0].numpy()
    return EquilibriumLens(
        dim=n_scalar_dir_from_sampled_dir.shape[1],
        n_pot_from_sampled_dir=n_pot_from_sampled_dir,
        tf_scalars_from_sampled_dir=tf_scalars_from_sampled_dir,
        tf_stat_from_sampled_dir=tf_stat_from_sampled_dir,
        tf_pot_stat=tf_pot_stat)

  def get_residual_symmetry(self, position, **kwargs):
    """Extracts the residual gauge symmetry."""
    del position, kwargs  # Not used by default implementation.
    return None

  def gravitino_masses_from_position(self, position, **kwargs):
    """Extracts gravitino-masses."""
    t_pos = mu.tff64(position)
    t_pot, *_ = self.tf_ext_sugra_tensors(t_pos, **kwargs)
    t_vielbein = self.tf_vielbein(t_pos)
    t_T = self.tf_T(t_vielbein, **kwargs)
    t_A1, *_ = self.tf_A123(t_T, want_A1=True, want_A2=False, want_A3=False)
    return self.gravitino_masses(t_A1.numpy(), t_pot.numpy())

  def get_physics(self, position, metadata, **kwargs):
    """Extracts information about the physics."""
    t_pos = mu.tff64(position)
    t_pot, *_, t_stat = self.tf_ext_sugra_tensors(t_pos, **kwargs)
    pot, stat = t_pot.numpy(), t_stat.numpy()
    t_vielbein = self.tf_vielbein(t_pos)
    t_T = self.tf_T(t_vielbein, **kwargs)
    ts_A123 = self.tf_A123(t_T, want_A1=True, want_A2=True, want_A3=True)
    t_A1, *_ = ts_A123
    mm_scalars = self.tf_scalar_massmatrix(t_pos, **kwargs)
    mass_eigenspaces_scalars = mu.gramian_eigenspaces(mm_scalars)
    mass_eigenspaces_fermions = []
    mass_eigenspaces_vectors = []
    mass_eigenspaces_gravitinos = []
    try:
      mass_eigenspaces_gravitinos = mu.gramian_eigenspaces(
          self.tf_gravitino_massmatrix(t_A1, t_pot).numpy())
    except Exception as exn:  # pylint:disable=broad-except
      # Could not compute Gravitino masses. This is non-fatal.
      if self._verbose:
        print('Gravitino masses problem:', repr(exn))
    try:
      mass_eigenspaces_fermions = mu.gramian_eigenspaces(
          self.tf_fermion_massmatrix(ts_A123, t_pot).numpy())
    except Exception as exn:  # pylint:disable=broad-except
      # Could not compute Fermion masses. This is non-fatal.
      if self._verbose:
        print('Fermion masses problem:', repr(exn))
    try:
      mass_eigenspaces_vectors = mu.gramian_eigenspaces(
          self.tf_vector_massmatrix(ts_A123, t_pot).numpy())
    except Exception as exn:  # pylint:disable=broad-except
      # Could not compute Vector masses. This is non-fatal.
      if self._verbose:
        print('Vector masses problem:', repr(exn))
    return EquilibriumPhysics(
        metadata=dict(metadata),  # Copy.
        potential=pot,
        stationarity=stat,
        position=numpy.array(position),
        residual_symmetry=self.get_residual_symmetry(position, **kwargs),
        num_spurious_vector_masses=self.signature.num_spurious_vector_masses,
        mass_eigenspaces_gravitinos=mass_eigenspaces_gravitinos,
        mass_eigenspaces_scalars=mass_eigenspaces_scalars,
        mass_eigenspaces_fermions=mass_eigenspaces_fermions,
        mass_eigenspaces_vectors=mass_eigenspaces_vectors)

  def show_position_text(self, position):
    """Returns a text-string that shows the position."""
    # Default implementation is slightly boring.
    return repr(position.round(5).tolist())

  def show_position_tex(self, position, digits=5):
    """Returns a text-string that shows the position."""
    # Default implementation is slightly boring.
    fmt_num = f'%+.{digits}f'
    phis = ', '.join(
        r'$\phi_{%d}=%s$' % (n, fmt_num % x)
        for n, x in enumerate(position)
        if abs(x) >= 0.1**digits)
    if not phis:
      phis = r'$\phi_k=0$'
    return (r'{\scriptstyle\begin{minipage}[t]{10cm}'
            r'\begin{flushleft}%s\end{flushleft}\end{minipage}}\\' % phis)

  def show_physics_text(self, equilibrium_physics,
                        susy_mass_threshold=1e-4):
    """Computes a human-readable string that describes a solution's physics."""
    p = equilibrium_physics
    pos_txt = self.show_position_text(p.position)
    susy_text = ', N=?'
    if p.mass_eigenspaces_gravitinos:
      num_susy = sum(space.shape[0]
                     for m, space in p.mass_eigenspaces_gravitinos
                     if m <= 1 + susy_mass_threshold)
      if num_susy:
        susy_text = f', N={num_susy}'
      else:
        susy_text = ''
    # The Breitenlohner-Freedman Bound.
    bf_bound = -(self.signature.dim - 1)**2 / 4
    bf_stable_text = ' BF=?'
    if p.mass_eigenspaces_scalars:
      # We have this extra num_susy criterion here as SUSY vacua are
      # automatically stable, but there are known examples where they
      # saturate the BF-bound.
      if num_susy or all(m >= bf_bound for m, _ in p.mass_eigenspaces_scalars):
        bf_stable_text = ', BF-stable'
      elif any(m < bf_bound -0.001 for m, _ in p.mass_eigenspaces_scalars):
        bf_stable_text = ', BF-unstable'
        # If there was no clear violation, keep the 'unclear' default.
    return (
        f'### P={"%.8f" % p.potential} '
        f'S={"%.3g" % p.stationarity} '
        f'{p.metadata} ###\n'
        #
        f'''dim(GG)={p.residual_symmetry.all_gens.shape[0]
                     if p.residual_symmetry else "???"}, '''
        f'''rank={p.residual_symmetry.rank
                     if p.residual_symmetry else "???"}, '''
        f'''num_u1s={p.residual_symmetry.u1_gens.shape[0]
                     if p.residual_symmetry else "???"}'''
        f'{susy_text}{bf_stable_text}\n\n'
        f'approx_pos={pos_txt}'
        f'{p.summarize_spectra_text()}\n')

  def show_physics_tex(self, equilibrium_physics,
                       tag_kwargs=default_tag_kwargs,
                       tex_templates=(_summary_solution_template_tex,
                                      _detailed_solution_template_tex),
                       susy_mass_threshold=1e-4):
    """Computes latex strings that describe a solution's physics."""
    p = equilibrium_physics
    text_tag_kwargs = {**dict(tag_kwargs), **dict(tag='sol')}
    text_tag = S_id(p.potential, **text_tag_kwargs)
    tex_tag = S_id(p.potential, **dict(tag_kwargs))
    def mathfrak(s):
      return re.sub('([a-zA-Z]+)', lambda m: r'\mathfrak{%s}' % m[1], s)
    symmetry_tex = ''
    if p.residual_symmetry is not None:
      if p.residual_symmetry.rank == 0:
        symmetry_tex = r'\emptyset'
      else:
        num_u1s = p.residual_symmetry.u1_gens.shape[0]
        u1s = '+'.join(
            [r'\mathfrak{u}(1)'] * num_u1s)
        dim_semisimple = p.residual_symmetry.semisimple_subgroup_gens.shape[0]
        semisimple = mathfrak(_SEMISIMPLE_BY_DIM_AND_RANK.get(
            (dim_semisimple, p.residual_symmetry.rank - num_u1s),
            ''))
        if dim_semisimple and not semisimple:
          semisimple = r'\{\text{%d-dim.\;semisimple}\}' % dim_semisimple
        symmetry_tex = '+'.join(filter(None, [semisimple, u1s]))
    symmetry_breaking_tex = (
        r'{}\rightarrow {}'.format(
            mathfrak(self.signature.gauge_algebra_name),
            symmetry_tex))
    susy_tex = r'{\mathcal N}=?'
    if p.mass_eigenspaces_gravitinos:
      num_susy = sum(space.shape[0]
                     for m, space in p.mass_eigenspaces_gravitinos
                     if m <= 1 + susy_mass_threshold)
      if num_susy:
        susy_tex = r'\mathcal{N}={%d}' % num_susy
      else:
        susy_tex = ''
    susy_long_tex = r', \quad ' + susy_tex if susy_tex else susy_tex
    bf_bound = -(self.signature.dim - 1)**2 / 4
    bf_stable_tex = r',\quad \text{BF=?}'
    bf_stable_yn_tex = '?'
    if p.mass_eigenspaces_scalars:
      # We have this extra num_susy criterion here as SUSY vacua
      # are automatically stable, but there are known examples
      # where they saturate the BF-bound.
      if num_susy or all(m >= bf_bound for m, _ in p.mass_eigenspaces_scalars):
        bf_stable_tex = r',\quad \text{BF-stable}'
        bf_stable_yn_tex = r'\checkmark'
      elif any(m < bf_bound -0.001 for m, _ in p.mass_eigenspaces_scalars):
        bf_stable_tex = r',\quad \text{BF-unstable}'
        bf_stable_yn_tex = ''
        # If there is no clear violation, keep the default 'unclear'.
    sqrt_stationarity = math.sqrt(p.stationarity + 1e-300)
    sqrt_stationarity_log10 = math.floor(math.log(sqrt_stationarity, 10))
    stationarity_short_tex = r'<{%.1f}\cdot10^{%d}' % (
        sqrt_stationarity * 10**(-sqrt_stationarity_log10) + 0.1,
        sqrt_stationarity_log10)
    stationarity_tex = r'|\nabla V/g^2|' + stationarity_short_tex
    tex_params = dict(
        text_tag=text_tag,
        tex_tag=tex_tag,
        symmetry_tex=symmetry_tex,
        symmetry_breaking_tex=symmetry_breaking_tex,
        susy_tex=susy_tex,
        susy_long_tex=susy_long_tex,
        bf_stable_tex=bf_stable_tex,
        bf_stable_yn_tex=bf_stable_yn_tex,
        stationarity_tex=stationarity_tex,
        stationarity_short_tex=stationarity_short_tex,
        potential=p.potential,
        spectra=p.summarize_spectra_tex(),
        position_tex=self.show_position_tex(p.position))
    return tuple(tex_template  % tex_params for tex_template in tex_templates)

  def tex_all_solutions(self, positions, out_stem,
                        tag_kwargs=default_tag_kwargs,
                        metadata=(),
                        canonicalize=True,
                        verbose=True,
                        **kwargs):
    """Produces LaTeX code for a collection of solutions."""
    out_summary_table = out_stem + '_summary.tex'
    out_detailed_table = out_stem + '_detailed.tex'
    mu.rm(out_summary_table)
    mu.rm(out_detailed_table)
    added_hline = False
    for num_solution, raw_params in enumerate(positions, 1):
      if canonicalize:
        params = self.canonicalize_equilibrium(raw_params)
      else:
        params = raw_params
      physics = self.get_physics(params, dict(metadata), **kwargs)
      summary_tex, detailed_tex = self.show_physics_tex(
          physics, tag_kwargs=tag_kwargs)
      with open(out_summary_table, 'at') as h_out:
        h_out.write(f'% ===\n{summary_tex}\n')
        if num_solution % 5 == 0:
          h_out.write('\\hline\n')
          added_hline = True
        else:
          added_hline = False
        h_out.flush()
      with open(out_detailed_table, 'at') as h_out:
        h_out.write(f'% ===\n{detailed_tex}\n')
        h_out.flush()
      if verbose:
        print(f'### V/g^2={physics.potential}, S={physics.stationarity:.6g}')
    if not added_hline:
      with open(out_summary_table, 'at') as h_out:
        h_out.write('\\hline\n')

  def refine_solution(self, position,
                      optimizer_strategies='GB',
                      potential_change_threshold=1e-6,
                      gtol=1e-12,
                      verbose=False,
                      **kwargs):
    """Basic recipe for refining the quality of a solution."""
    current_position = position
    current_pot, current_stat = self.potential_and_stationarity(
        current_position, **kwargs)
    for strategy in optimizer_strategies:
      new_pot, new_stat, _ = new_info = self.find_equilibrium(
          current_position,
          minimize_kwargs=dict(
              gtol=gtol,
              mdnewton_maxsteps=2,
              gradient_steps=((100, 1e-6),),
              strategy=strategy),
          **kwargs)
      success = (new_stat < current_stat and
                 abs(new_pot - current_pot) <= potential_change_threshold)
      if verbose:
        if success:
          print('Step %r %ssuccessful, P: %.8f -> %.8f, S: %.6g -> %.6g' % (
              strategy, '' if success else 'un',
              current_pot, new_pot,
              current_stat, new_stat))
      if success:
        current_pot, current_stat, current_position = new_info
    return current_pot, current_stat, current_position
