function Sphere(center, color)

if nargin < 2
  color = [0.65 0.65 0.75];
end
radius = 0.01; 
pts = 50;
[X, Y, Z] = sphere(pts);

  
Xn = radius*X + center(1);
Yn = radius*Y + center(2);
Zn = radius*Z + center(3);
  
surf(Xn,Yn,Zn, 'FaceColor', color)
end