% Copyright 2019 Google LLC
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function domain = TrotStance1(model, load_path, bezier_deg)
% Trot Stance One
% Contact: FrontRightToe and BackLeftToe

%% first make a copy of the robot model
%| @note Do not directly assign with the model variable, since it is a
%handle object.
domain = copy(model);
% set the name of the new copy
domain.setName('TrotStance1');

% Extract state variables
x = domain.States.x;
dx = domain.States.dx;

% define frictional properties
fric_coef.mu = 0.6;
fric_coef.gamma = 100;

% add contact
frontright_toe = ToContactFrame(domain.ContactPoints.FrontRightToe, ...
  'PointContactWithFriction');
% load symbolic expressions for contact (holonomic) constraints
domain = addContact(domain, frontright_toe, fric_coef, [], load_path);

backleft_toe = ToContactFrame(domain.ContactPoints.BackLeftToe, ...
  'PointContactWithFriction');
% load symbolic expressions for contact (holonomic) constraints
domain = addContact(domain, backleft_toe, fric_coef, [], load_path);

% add event
% height of non-stance foot (frontleft toe)
pFL_nsf = getCartesianPosition(domain, domain.ContactPoints.FrontLeftToe);
hFL_nsf = UnilateralConstraint(domain, pFL_nsf(3), 'frontleftFootHeight', 'x');

% height of non-stance foot (backright toe)
pBR_nsf = getCartesianPosition(domain, domain.ContactPoints.BackRightToe);
hBR_nsf = UnilateralConstraint(domain, pBR_nsf(3), 'backrightFootHeight', 'x');
domain = addEvent(domain, [hFL_nsf, hBR_nsf]);

% phase variable: time
t = SymVariable('t');
p = SymVariable('p', [2, 1]);
tau = (t - p(2)) / (p(1) - p(2));

%% Adding relative degree two holonomic outputs:
%
y_FL_swg = x('motor_front_leftL_joint');
y_BL_swg = x('motor_back_leftL_joint');
y_FR_swg = x('motor_front_rightR_joint');
y_BR_swg = x('motor_back_rightR_joint');
y_FL_ext = x('knee_front_leftL_joint');
y_BL_ext = x('knee_back_leftL_joint');
y_FR_ext = x('knee_front_rightR_joint');
y_BR_ext = x('knee_back_rightR_joint');


ya = [y_FL_swg; y_BL_swg; y_FR_swg; y_BR_swg; ...
  y_FL_ext; y_BL_ext; y_FR_ext; y_BR_ext];

ya_label = {'frontleft_swg', 'backleft_swg', 'frontright_swg', ...
  'backright_swg', 'frontleft_ext', 'backleft_ext', 'frontright_ext', ...
  'backright_ext'};

hol_idx = [1, 2, 3, 4, 5, 6, 7, 8];

% assign virtual constraints
y_hol = VirtualConstraint(domain, ya(hol_idx), 'time', 'DesiredType', ...
  'Bezier', 'PolyDegree', bezier_deg, 'RelativeDegree', 2, 'OutputLabel', ...
  {ya_label(hol_idx)}, 'PhaseType', 'TimeBased', 'PhaseVariable', tau, ...
  'PhaseParams', p, 'Holonomic', true, 'LoadPath', load_path);

domain = addVirtualConstraint(domain, y_hol);


end