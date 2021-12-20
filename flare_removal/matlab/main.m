%  // clang-format off
clear
close all

%% Typical parameters for a smartphone camera.
% Nominal wavelength (m).
lambda = 550e-9;
% Focal length (m).
f = 2.2e-3;
% Pixel pitch on the sensor (m).
delta = 1e-6;
% Sensor size (width & height, m).
l = 6e-3;
% Simulation resolution, in both spatial and frequency domains.
res = l / delta;

%% Compute defocus phase shift and aperture mask in the Fourier domain.
% Frequency range (extent) of the Fourier transform (m ^ -1).
lf = lambda * f / delta;
% Diameter of the circular low-pass filter on the Fourier plane.
df = 1e-3;
% Low-pass radius, normalized by simulation resolution.
rf_norm = df / 2 / lf;
[defocus_phase, aperture_mask] = GetDefocusPhase(res, rf_norm);

%% Wavelengths at which the spectral response is sampled.
num_wavelengths = 73;
wavelengths = linspace(380, 740, num_wavelengths) * 1e-9;

%% Create output directories.
out_dir = 'streaks/';
mkdir(out_dir);
aperture_dir = 'apertures/';
mkdir(aperture_dir);
out_crop = 800;

%% generate the PSFs
parfor tt = 1:1000
  aperture = RandomDirtyAperture(aperture_mask);
  imwrite(aperture, strcat(aperture_dir, sprintf('%03d.png',tt - 1)));

  %% Random RGB spectral response.
  wl_to_rgb = RandomSpectralResponse(wavelengths).';

  for ii = 1:4
    %% Random defocus.
    defocus_crop = 4000;
    defocus = randn * 5;
    psf_rgb = GetPsf(aperture, defocus_phase * defocus, ...
                     wavelengths ./ lambda, wl_to_rgb, defocus_crop);

    for kk = 1:4
      %% Randomly crop and distort the PSF.
      focal_length_px = f / delta * [1, 1];
      sensor_crop = [2400, 2400];
      principal_point = sensor_crop / 2;
      radial_distortion = [randn * 0.8, 0];
      camera_params = cameraIntrinsics( ...
          focal_length_px, principal_point, sensor_crop, ...
          'RadialDistortion', radial_distortion);
      psf_cropped = CropRandom(psf_rgb, sensor_crop);
      psf_distorted = undistortImage(psf_cropped, camera_params);

      %% Apply global tone curve (gamma) and write to disk.
      psf_ds = imresize(psf_distorted, 0.5, 'box');
      psf_out = EqualizeChannels(CropCenter(psf_ds, out_crop));
      psf_gamma = abs(psf_out .^ (1/2.2));
      psf_gamma = min(psf_gamma, 2^16 - 1);
      psf_u16 = uint16(psf_gamma);

      output_file_name = sprintf('aperture%04d_blur%02d_crop%02d.png', ...
                                 tt - 1, ii - 1, kk - 1);
      imwrite(psf_u16, strcat(out_dir, output_file_name));
      fprintf('Written to disk: %s\n', output_file_name);

    end
  end
end
