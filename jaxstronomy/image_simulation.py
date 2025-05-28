# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Strong lensing image simulation pipeline in jax.

This module includes functions to map from a strong lensing configuration to the
corresponding observed image using jax-optimized functions. Inspiration taken
from lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

import functools
import itertools
from typing import Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from jaxstronomy import cosmology_utils
from jaxstronomy import lens_models
from jaxstronomy import psf_models
from jaxstronomy import source_models
from jaxstronomy import utils

LENS_MODELS = tuple(
    [lens_models.__getattribute__(model) for model in lens_models.__all__])
LENS_MODEL_PARAMETERS = tuple([model.parameters for model in LENS_MODELS])
ALL_LENS_MODEL_PARAMETERS = tuple(
    sorted(set(itertools.chain(*LENS_MODEL_PARAMETERS))))
SOURCE_MODELS = tuple(
    [source_models.__getattribute__(model) for model in source_models.__all__])
SOURCE_MODEL_PARAMETERS = tuple([model.parameters for model in SOURCE_MODELS])
ALL_SOURCE_MODEL_PARAMETERS = tuple(
    sorted(set(itertools.chain(*SOURCE_MODEL_PARAMETERS))))
PSF_MODELS = tuple(
    [psf_models.__getattribute__(model) for model in psf_models.__all__])
PSF_MODEL_PARAMETERS = tuple([model.parameters for model in PSF_MODELS])
ALL_PSF_MODEL_PARAMETERS = tuple(
    sorted(set(itertools.chain(*PSF_MODEL_PARAMETERS))))


def generate_image(
    grid_x,
    grid_y,
    kwargs_lens_all,
    kwargs_source_slice,
    kwargs_lens_light_slice,
    kwargs_psf,
    cosmology_params,
    z_lens_array,
    z_source,
    kwargs_detector,
):
  """Generate an image given the source, lens light, and mass profiles.

  Args:
    grid_x: X-coordinates of grid on which image will be generated.
    grid_y: Y-coordinates of grid on which image will be generated.
    kwargs_lens_all: Keyword arguments for each lens model. The kwargs for
      should contain keyword arguments to pass to derivative functions for each
      lens model. Keys should include parameters for all lens models (due to use
      of `jax.lax.switch`) and `model_index` which defines the model to pass the
      parameters to. Parameter values not relevant for the specified model are
      discarded. Values should be of type jnp.ndarray and have length equal to
      the total number of lens models. For more detailed discussion see
      documentation of `jax.lax.scan`, which is used to iterate over the models.
    kwargs_source_slice: Keyword arguments to pass to brightness functions for
      each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to.
    kwargs_lens_light_slice: Keyword arguments to pass to brightness functions
      for each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to.
    kwargs_psf: Keyword arguments defining the point spread function. The psf is
      applied in the supersampled space, so the size of pixels should be defined
      with respect to the supersampled space.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens_array: Redshifts for each lens model.
    z_source: Redshift of the source.
    kwargs_detector: Keyword arguments defining the detector configuration.

  Returns:
    Image after gravitational lensing at supersampling resolution. For
    consistency with lenstronomy, units are per pixel area of the detector, not
    the supersampling resolution.

  Notes:
    kwargs_detector must be passed in through functools.partial or equivalent if
    you want to jit compile this function.
  """
  image_array = source_surface_brightness(grid_x, grid_y, kwargs_lens_all,
                                          kwargs_source_slice, kwargs_detector,
                                          cosmology_params, z_lens_array,
                                          z_source)
  image_array += lens_light_surface_brightness(grid_x, grid_y,
                                               kwargs_lens_light_slice,
                                               kwargs_detector)
  image = jnp.reshape(
      image_array,
      (kwargs_detector['n_x'] * kwargs_detector['supersampling_factor'],
       kwargs_detector['n_y'] * kwargs_detector['supersampling_factor']))

  return psf_convolution(image, kwargs_psf)


def psf_convolution(
    image,
    kwargs_psf,
):
  """Convolve image with the point spread function.

  Args:
    image: Image to convolve
    kwargs_psf: Keyword arguments defining the point spread function.

  Returns:
    Convolved image.
  """
  # Psf not accounting for supersampling by default is prone to user error.
  # Consider changing.
  psf_functions = [model.convolve for model in PSF_MODELS]
  return jax.lax.switch(kwargs_psf['model_index'], psf_functions, image,
                        kwargs_psf)


def noise_realization(
    image, rng,
    kwargs_detector):
  """Return noise realization for input image.

  Args:
    image: Image from which to draw poisson noise.
    rng: jax PRNG key used for noise realization.
    kwargs_detector: Keyword arguments defining the detector configuration.

  Returns:
    Noise realization for image.
  """
  rng_normal, rng_poisson = jax.random.split(rng)

  # Calculate expected background noise from detector kwargs.
  exposure_time_total = (
      kwargs_detector['exposure_time'] * kwargs_detector['num_exposures'])
  read_noise = (
      kwargs_detector['read_noise']**2 * kwargs_detector['num_exposures'])
  sky_brightness_cps = utils.magnitude_to_cps(
      kwargs_detector['sky_brightness'],
      kwargs_detector['magnitude_zero_point'])
  sky_brightness_tot = (
      exposure_time_total * sky_brightness_cps *
      kwargs_detector['pixel_width']**2)
  background_noise = (
      jnp.sqrt(read_noise + sky_brightness_tot) / exposure_time_total)

  # By default all simulations are done in units of counts per second, but you
  # want to calculate poisson statistics in units of counts.
  flux_noise = jnp.sqrt(jax.nn.relu(image) / exposure_time_total)

  noise = jax.random.normal(rng_normal, image.shape) * background_noise
  noise += jax.random.normal(rng_poisson, image.shape) * flux_noise
  return noise


def lens_light_surface_brightness(
    theta_x, theta_y,
    kwargs_lens_light_slice,
    kwargs_detector):
  """Return the lens light surface brightness.

  Args:
    theta_x: X-coordinates in angular units.
    theta_y: Y-coordinates in angular units.
    kwargs_lens_light_slice: Keyword arguments to pass to brightness functions
      for each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to.
    kwargs_detector: Keyword arguments defining the detector configuration.

  Returns:
    Surface brightness of lens light as 1D array.
  """
  lens_light_flux = _surface_brightness(theta_x, theta_y,
                                        kwargs_lens_light_slice)
  return lens_light_flux * kwargs_detector['pixel_width']**2


def source_surface_brightness(
    alpha_x,
    alpha_y,
    kwargs_lens_all,
    kwargs_source_slice,
    kwargs_detector,
    cosmology_params,
    z_lens_array,
    z_source,
):
  """Return the lensed source surface brightness.

  Args:
    alpha_x: Initial x-component of deflection at each position.
    alpha_y: Initial y-component of deflection at each position.
    kwargs_lens_all: Keyword arguments for each lens model. The kwargs for
      should contain keyword arguments to pass to derivative functions for each
      lens model. Keys should include parameters for all lens models (due to use
      of `jax.lax.switch`) and `model_index` which defines the model to pass the
      parameters to. Parameter values not relevant for the specified model are
      discarded. Values should be of type jnp.ndarray and have length equal to
      the total number of lens models. For more detailed discussion see
      documentation of `jax.lax.scan`, which is used to iterate over the models.
    kwargs_source_slice: Keyword arguments to pass to brightness functions for
      each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to.
    kwargs_detector: Keyword arguments defining the detector configuration. This
      includes potential supersampling in the lensing calculation.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens_array: Redshifts for each lens model.
    z_source: Redshift of the source.

  Returns:
    Lensed source surface brightness as 1D array.
  """
  image_flux_array = _image_flux(alpha_x, alpha_y, kwargs_lens_all,
                                 kwargs_source_slice, cosmology_params,
                                 z_lens_array, z_source)
  # Scale by pixel area to go from flux to surface brightness.
  return image_flux_array * kwargs_detector['pixel_width']**2


def _image_flux(alpha_x, alpha_y,
                kwargs_lens_all,
                kwargs_source_slice,
                cosmology_params,
                z_lens_array, z_source):
  """Calculate image flux after ray tracing onto the source.

  Args:
    alpha_x: Initial x-component of deflection at each position.
    alpha_y: Initial y-component of deflectoin at each position.
    kwargs_lens_all: Keyword arguments for each lens model. The kwargs for
      should contain keyword arguments to pass to derivative functions for each
      lens model. Keys should include parameters for all lens models (due to use
      of `jax.lax.switch`) and `model_index` which defines the model to pass the
      parameters to. Parameter values not relevant for the specified model are
      discarded. Values should be of type jnp.ndarray and have length equal to
      the total number of lens models. For more detailed discussion see
      documentation of `jax.lax.scan`, which is used to iterate over the models.
    kwargs_source_slice: Keyword arguments to pass to brightness functions for
      each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to. Parameter values not relevant for the specified
      model are discarded. Values should be of type jnp.ndarray and have length
      equal to the total number of light models in the slice. For more detailed
      discussion see documentation of `jax.lax.scan`, which is used to iterate
      over the models.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens_array: Redshifts for each lens model.
    z_source: Redshift of the source.

  Returns:
    Image flux.
  """
  x_source_comv, y_source_comv = _ray_shooting(alpha_x, alpha_y,
                                               kwargs_lens_all,
                                               cosmology_params, z_lens_array,
                                               z_source)
  x_source, y_source = cosmology_utils.comoving_to_angle(
      x_source_comv, y_source_comv, cosmology_params, z_source)
  return _surface_brightness(x_source, y_source, kwargs_source_slice)


def _ray_shooting(
    alpha_x,
    alpha_y,
    kwargs_lens_all,
    cosmology_params,
    z_lens_array,
    z_source,
):
  """Ray shoot over all of the redshift slices between observer and source.

  Args:
    alpha_x: Initial x-component of deflection at each position.
    alpha_y: Initial y-component of deflectoin at each position.
    kwargs_lens_all: Keyword arguments for each lens model. The kwargs for
      should contain keyword arguments to pass to derivative functions for each
      lens model. Keys should include parameters for all lens models (due to use
      of `jax.lax.switch`) and `model_index` which defines the model to pass the
      parameters to. Parameter values not relevant for the specified model are
      discarded. Values should be of type jnp.ndarray and have length equal to
      the total number of lens models. For more detailed discussion see
      documentation of `jax.lax.scan`, which is used to iterate over the models.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens_array: Redshifts for each lens model.
    z_source: Redshift of the source.

  Returns:
    Comoving x- and y-coordinate after ray shooting.
  """
  # Initially all our light rays are localized at the observer.
  comv_x = jnp.zeros_like(alpha_x)
  comv_y = jnp.zeros_like(alpha_y)

  z_lens_last = 0.0

  # The state to pass to scan.
  state = (comv_x, comv_y, alpha_x, alpha_y, z_lens_last)

  ray_shooting_step = functools.partial(
      _ray_shooting_step, cosmology_params=cosmology_params, z_source=z_source)

  # Scan over all of the lens models in our system to calculate deflection and
  # ray shoot between lens models.
  final_state, _ = jax.lax.scan(ray_shooting_step, state, {
      'z_lens': z_lens_array,
      'kwargs_lens': kwargs_lens_all
  })

  comv_x, comv_y, alpha_x, alpha_y, z_lens_last = final_state

  # Continue the ray tracing until the source.
  delta_t = cosmology_utils.comoving_distance(cosmology_params, z_lens_last,
                                              z_source)
  comv_x, comv_y = _ray_step_add(comv_x, comv_y, alpha_x, alpha_y, delta_t)

  return comv_x, comv_y


def _ray_shooting_step(
    state,
    kwargs_z_lens,
    cosmology_params,
    z_source,
):
  """Conduct ray shooting between two lens models.

  Args:
    state: The current comoving positions, deflections, and the previous lens
      model redshift.
    kwargs_z_lens: Dict with keys `z_lens`, the redshift of the next lens model,
      and 'kwargs_lens`, the keyword arguments specifying the next lens model
      (through `model_index` value) and the lens model parameters. Due to the
      requirements of `jax.lax.switch`, `kwargs_lens` must have a key-value pair
      for all lens models included in `lens_models.py`, even if those lens
      models will not be used.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_source: Redshift of the source.

  Returns:
    Two copies of the new state, which is a tuple of new comoving positions, new
    deflection, and
    the redshift of the current lens.

  Notes:
    For use with `jax.lax.scan`.
  """
  comv_x, comv_y, alpha_x, alpha_y, z_lens_last = state

  # The displacement from moving along the deflection direction.
  delta_t = cosmology_utils.comoving_distance(cosmology_params, z_lens_last,
                                              kwargs_z_lens['z_lens'])
  comv_x, comv_y = _ray_step_add(comv_x, comv_y, alpha_x, alpha_y, delta_t)
  alpha_x, alpha_y = _add_deflection(comv_x, comv_y, alpha_x, alpha_y,
                                     kwargs_z_lens['kwargs_lens'],
                                     cosmology_params, kwargs_z_lens['z_lens'],
                                     z_source)

  new_state = (comv_x, comv_y, alpha_x, alpha_y, kwargs_z_lens['z_lens'])

  # Second return is required by scan, but will be ignored by the compiler.
  return new_state, new_state


def _ray_step_add(comv_x, comv_y,
                  alpha_x, alpha_y,
                  delta_t):
  """Conduct ray step by adding deflections to comoving positions.

  Args:
    comv_x: Comoving x-coordinate.
    comv_y: Comoving y-coordinate.
    alpha_x: Current physical x-component of deflection at each position.
    alpha_y: Current physical y-component of deflection at each position.
    delta_t: Comoving distance between current slice and next slice.

  Returns:
    Updated x- and y-coordinate in comoving Mpc.
  """
  return comv_x + alpha_x * delta_t, comv_y + alpha_y * delta_t


def _add_deflection(comv_x, comv_y,
                    alpha_x, alpha_y,
                    kwargs_lens,
                    cosmology_params,
                    z_lens,
                    z_source):
  """Calculate the deflection for a specific lens model.

  Args:
    comv_x: Comoving x-coordinate.
    comv_y: Comoving y-coordinate.
    alpha_x: Current physical x-component of deflection at each position.
    alpha_y: Current physical y-component of deflection at each position.
    kwargs_lens: Keyword arguments specifying the model (through `model_index`
      value) and the lens model parameters. Due to the nature of
      `jax.lax.switch` this must have a key-value pair for all lens models
      included in `lens_models.py`, even if those lens models will not be used.
      `model_index` of -1 indicates that the previous total should be returned.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens: Redshift of the slice (i.e. current redshift).
    z_source: Redshift of the source.

  Returns:
    New x- and y-component of the deflection at each specified position.
  """
  theta_x, theta_y = cosmology_utils.comoving_to_angle(comv_x, comv_y,
                                                       cosmology_params, z_lens)

  # All of our derivatives are defined in reduced coordinates.
  alpha_x_reduced, alpha_y_reduced = _calculate_derivatives(
      kwargs_lens, theta_x, theta_y)

  alpha_x_update = cosmology_utils.reduced_to_physical(alpha_x_reduced,
                                                       cosmology_params, z_lens,
                                                       z_source)
  alpha_y_update = cosmology_utils.reduced_to_physical(alpha_y_reduced,
                                                       cosmology_params, z_lens,
                                                       z_source)

  return alpha_x - alpha_x_update, alpha_y - alpha_y_update


def _calculate_derivatives(
    kwargs_lens,
    theta_x,
    theta_y,
):
  """Calculate the derivatives for the specified lens model.

  Args:
    kwargs_lens: Keyword arguments specifying the model (through `model_index`
      value) and the lens model parameters. Due to the nature of
      `jax.lax.switch` this must have a key-value pair for all lens models
      included in `lens_models.py`, even if those lens models will not be used.
      `model_index` of -1 indicates that the previous total should be returned.
    theta_x: X-coordinate at which to evaluate the derivative.
    theta_y: Y-coordinate at which to evaluate the derivative.

  Returns:
    Change in x- and y-component of derivative caused by lens model. X- and
    y-component stacked along axis=0.
  """
  # The jax.lax.switch compilation requires that all of the functions take the
  # same inputs. We accomplish this using our wrapper and picking out only the
  # parameters required for that model.
  derivative_functions = [
      utils.unpack_parameters_xy(LENS_MODELS[model_index].derivatives,
                                 LENS_MODEL_PARAMETERS[model_index])
      for model_index in range(len(LENS_MODELS))
  ]

  # Condition on model_index to allow for padding.
  def calculate_derivative():
    alpha_x, alpha_y = jax.lax.switch(kwargs_lens['model_index'],
                                      derivative_functions, theta_x, theta_y,
                                      kwargs_lens)
    return [alpha_x, alpha_y]

  def identity():
    return [jnp.zeros_like(theta_x), jnp.zeros_like(theta_y)]

  return jax.lax.cond(kwargs_lens['model_index'] == -1, identity,
                      calculate_derivative)


def _surface_brightness(
    theta_x, theta_y,
    kwargs_source_slice):
  """Return the surface brightness for a slice of light models.

  Args:
    theta_x: X-coordinate at which to evaluate the surface brightness.
    theta_y: Y-coordinate at which to evaluate the surface brightness.
    kwargs_source_slice: Keyword arguments to pass to brightness functions for
      each light model. Keys should include parameters for all source models
      (due to use of `jax.lax.switch`) and `model_index` which defines the model
      to pass the parameters to. Parameter values not relevant for the specified
      model are discarded. Values should be of type jnp.ndarray and have length
      equal to the total number of light models in the slice. For more detailed
      discussion see documentation of `jax.lax.scan`, which is used to iterate
      over the models.

  Returns:
    Surface brightness summed over all sources.
  """
  add_surface_brightness = functools.partial(
      _add_surface_brightness, theta_x=theta_x, theta_y=theta_y)
  brightness_total = jnp.zeros_like(theta_x)
  brightness_total, _ = jax.lax.scan(add_surface_brightness, brightness_total,
                                     kwargs_source_slice)
  return brightness_total


def _add_surface_brightness(prev_brightness,
                            kwargs_source,
                            theta_x,
                            theta_y):
  """Return the surface brightness for a single light model.

  Args:
    prev_brightness: Previous brightness.
    kwargs_source: Kwargs to evaluate the source surface brightness.
    theta_x: X-coordinate at which to evaluate the surface brightness.
    theta_y: Y-coordinate at which to evaluate the surface brightness.

  Returns:
    Surface brightness of the source at given coordinates.
  """
  source_functions = [
      utils.unpack_parameters_xy(SOURCE_MODELS[model_index].function,
                                 SOURCE_MODEL_PARAMETERS[model_index])
      for model_index in range(len(SOURCE_MODELS))
  ]
  brightness = jax.lax.switch(kwargs_source['model_index'], source_functions,
                              theta_x, theta_y, kwargs_source)

  return prev_brightness + brightness, brightness
