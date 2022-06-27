%  // clang-format off
function cropped = CropRandom(im, crop)

if length(crop) == 1
  crop = [crop, crop];
end
window = randomCropWindow2d(size(im), crop);
cropped = imcrop(im, window);

end
