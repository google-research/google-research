%  // clang-format off
function cropped = CropCenter(im, crop)

if length(crop) == 1
  crop = [crop, crop];
end
window = centerCropWindow2d(size(im), crop);
cropped = imcrop(im, window);

end
