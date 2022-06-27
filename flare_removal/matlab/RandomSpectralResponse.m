%  // clang-format off
function response_matrix = RandomSpectralResponse(wavelengths)
% RandomSpectralResponse Random RGB spectral response matrix.
%
% response_matrix = RandomSpectralResponse(wavelengths)
% Returns the RGB response coefficients at the given wavelengths, with 
% reasonable random perturbations to simulate the uncertainty in a real-world 
% imaging system.
%
% Arguments
%
% wavelengths: an N-vector specifying N wavelengths where the spectral response
%              is sampled. Unit: meters.
%
% Returns
%
% response_matrix: a 3 x N matrix where each column contains the RGB response 
%                  factors for the corresponding wavelength in the input 
%                  argument.
%
% Required toolboxes: none.

rgb_centers = [620; 540; 460] * 1e-9 + rand([3, 1]) * 20e-9;
passband = 50e-9 + rand([3, 1]) * 10e-9;

r_response = EvaluateGaussian(wavelengths, rgb_centers(1), passband(1));
g_response = EvaluateGaussian(wavelengths, rgb_centers(2), passband(2));
b_response = EvaluateGaussian(wavelengths, rgb_centers(3), passband(3));

response_matrix = [r_response; g_response; b_response];

end

function val = EvaluateGaussian(x, mu, sigma)
val = exp(-(x - mu) .^ 2 / (2 * sigma .^ 2));
end
