%  // clang-format off
function equalized = EqualizeChannels(im)
% EqualizeChannels Equalizes channel means.
%
% equalized = EqualizeChannels(im)
% Applies scaling to each channel individually, such that the resulting mean
% value of each channel is the same as the minimum channel. This can be thought
% of as a white balance in some sense. Note that we apply a gain that's < 1
% here, which may introduce color artifacts if any input pixel is clipped due to
% saturation.
%
% Arguments:
%
% im: An [H, W, C]-array where C is the channel dimension.
%
% Returns:
%
% equalized: Same shape and type as `im`, with channel means equalized.
%
% Required toolboxes: none.

channel_means = mean(im, [1, 2]);
channel_gains = min(channel_means) / channel_means;
equalized = im .* channel_gains;

end
