%  // clang-format off
function [phase, mask] = GetDefocusPhase(n, r)
% GetDefocusPhase Phase shift due to defocus for a round aperture.
%
% [phase, mask] = GetDefocusPhase(n, aperture_r)
% Computes the phase shift per unit defocus in the Fourier domain. Also returns
% the corresponding circular mask on the Fourier plane that defines the valid
% region of the frequency response.
%
% Arguments
%
% n: Number of samples in each direction for the image and spectrum. The output
%    will be an [n, n]-array.
%
% r: Radius of the circular low-pass filter applied on the spectrum, assuming 
%    the spectrum is a unit square.
%
% Returns
%
% phase: Amount of (complex) phase shift in the spectrum for each unit (1) of
%        defocus. Zero outside the disk of radius `r`. [n, n]-array.
%
% mask: A centered disk of 1 surrounded by 0, representing the low-pass filter
%       that is applied to the spectrum (including the `phase` array above).
%       [n, n]-array.
%
% Required toolboxes: none.

%% Pixel center coordinates in Cartesian and polar forms.
sample_x = linspace(-(n - 1) / 2, (n - 1) / 2, n) / n / r;
[xx, yy] = meshgrid(sample_x);
[~, rr] = cart2pol(xx, yy);

%% The mask is simply a centered unit disk.
% Zernike polynomials below are only defined on the unit disk.
mask = rr <= 1;

%% Compute the Zernike polynomial of degree 2, order 0. 
% Zernike polynomials form a complete, orthogonal basis over the unit disk. The 
% "degree 2, order 0" component represents defocus, and is defined as (in 
% unnormalized form):
%
%     Z = 2 * r^2 - 1.
%
% Reference:
% Paul Fricker (2021). Analyzing LASIK Optical Data Using Zernike Functions.
% https://www.mathworks.com/company/newsletters/articles/analyzing-lasik-optical-data-using-zernike-functions.html
phase = single(2 * rr .^ 2 - 1);
phase(~mask) = 0;

end
