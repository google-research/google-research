======================================================================
jaxstronomy - Implementation of Strong Lensing Simulation Code in jax.
======================================================================

This package implements multiplane lensing in jax in order to generate realistic images of strong gravitational lenses.

Features
--------

The following are all implemented using jax.numpy and jax.jit compatible functions:

* Cosmology and Power Spectrum calculations.
* Lens models for Elliptical Power Law, NFW, Shear, and Truncated NFW profiles
* Source models for Interpolated and Elliptical Sersic light models
* Pixelated and Gaussian PSF models
* High-level ray-tracing calculations to map from input geometry to generated lens image.