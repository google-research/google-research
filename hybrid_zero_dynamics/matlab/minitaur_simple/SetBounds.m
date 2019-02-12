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

%% Set bounds for optimization problem
floatDOF = 6;
netDOF = 8;
totalDOF = length(minitaur.States.x);

num_hol = 8; % number of holonomic outputs
num_nonhol = 0; %number of nonholonomic outputs

%[7:10]
hip_joint_indices = floatDOF + linspace(1, netDOF/2, netDOF/2);
%[11:14]
knee_joint_indices = floatDOF + (netDOF / 2) + linspace(1, netDOF/2, netDOF/2);

hip_joint_indices_R = floatDOF + [3, 4];
knee_joint_indices_R = floatDOF + (netDOF / 2) + [3, 4];

hip_joint_indices_L = floatDOF + [1, 2];
knee_joint_indices_L = floatDOF + (netDOF / 2) + [1, 2];

%% add torque limits
umin_hip = -2.0;
umax_hip = -umin_hip;

umin_knee = -40;
umax_knee = -umin_knee;

%% Here, degree => number of coefficients viz. degree + 1
degree = BEZIER_DEG + 1;

%% joint limits

%% vee config
% %% Assume theta = 2*phi => shin is twice of thigh in length

%% ext: bullet => robot => frost

%% ext: pi rads => 0.3225 cms => 0

%% ext: pi/2 rads => 0.1882 cms => -0.1343

%% ext: 0 => 0.1581 cms => -0.1644

l1 = 0.1036;
l2 = 0.216;
emax = l1 + l2;
emin = l2 - l1;
ext = 0.024;
foot_dist = emax + ext;
max_ext = ext_rad2cm(pi/2+pi/6); % RSS Paper Limits PI/2 - 0.5;
min_ext = ext_rad2cm(pi/2-pi/6); % UB PI/2 + 0.5


theta_range = deg2rad([150, 210]); % RSS Paper PI + (-0.5 , 0.5)
knee_range = [min_ext, max_ext];

mtr_lb = theta_range(1);
mtr_ub = theta_range(2);
kne_lb = knee_range(1);
kne_ub = knee_range(2);

% Motors IDX: FL BL FR BR (Bullet Leg Order)
theta0 = 180;
gamma = (STEPLEN_DES / 0.2) * (30);
knee0 = ext_rad2cm(pi/2+(deg2rad(-30+2*gamma)));

signRL = [-1; -1; 1; 1];

sign_trot1 = [1; -1; -1; 1];
sign_trot2 = -sign_trot1;


% Set Bounds
model_bounds = minitaur.getLimits();

% positive angles
model_bounds.states.x.lb(hip_joint_indices_R) = deal(mtr_lb);
model_bounds.states.x.ub(hip_joint_indices_R) = deal(mtr_ub);

% negative angles
model_bounds.states.x.lb(hip_joint_indices_L) = deal(-mtr_ub);
model_bounds.states.x.ub(hip_joint_indices_L) = deal(-mtr_lb);

% Knee limits
model_bounds.states.x.lb(knee_joint_indices) = deal(kne_lb);
model_bounds.states.x.ub(knee_joint_indices) = deal(kne_ub);

model_bounds.states.x.x0 = zeros(totalDOF, 1);
model_bounds.states.x.x0(3) = 0.24;

model_bounds.states.x.x0(hip_joint_indices) = deg2rad(signRL*theta0);
model_bounds.states.x.x0(knee_joint_indices) = knee0 * ones(4, 1);
model_bounds.states.dx.x0 = deal(0);
model_bounds.states.ddx.x0 = deal(0);

%% relaxed state bounds
% model_bounds.states.x.lb = deal(-10);
% model_bounds.states.x.ub = deal(10);

%% Constraints bounds


model_bounds.constraints.average_steplen.desired = STEPLEN_DES;
model_bounds.constraints.average_steplen.lb = ...
  max(0, STEPLEN_DES-STEPLEN_MARGIN);
model_bounds.constraints.average_steplen.ub = STEPLEN_DES + STEPLEN_MARGIN;
model_bounds.constraints.average_steplen.x0 = STEPLEN_DES;
model_bounds.constraints.min_swingfoot_lift.desired = 0.03;


% Trot Stance One

bounds.TrotStance1 = model_bounds;

bounds.TrotStance1.time.t0.lb = 0;
bounds.TrotStance1.time.t0.ub = 0;
bounds.TrotStance1.time.t0.x0 = 0;

bounds.TrotStance1.time.tf.lb = 0.30;
bounds.TrotStance1.time.tf.ub = 1.20;
bounds.TrotStance1.time.tf.x0 = 0.55;

bounds.TrotStance1.time.duration.lb = 0.30;
bounds.TrotStance1.time.duration.ub = 1.20;
bounds.TrotStance1.time.duration.x0 = 0.55;

bounds.TrotStance1.states.x.x0(hip_joint_indices) = ...
  deg2rad(signRL*theta0+sign_trot1*gamma);
bounds.TrotStance1.states.x.x0(knee_joint_indices) = knee0 * ones(4, 1);

bounds.TrotStance1.inputs.Control.u.lb = [umin_hip * ones(netDOF/2, 1); ...
  umin_knee * ones(netDOF/2, 1)];
bounds.TrotStance1.inputs.Control.u.ub = [umax_hip * ones(netDOF/2, 1); ...
  umax_knee * ones(netDOF/2, 1)];
bounds.TrotStance1.inputs.Control.u.x0 = zeros(netDOF, 1);

bounds.TrotStance1.inputs.ConstraintWrench.fFrontRightToe.lb = ...
  -5000 * ones(3, 1);
bounds.TrotStance1.inputs.ConstraintWrench.fFrontRightToe.ub = ...
  5000 * ones(3, 1);
bounds.TrotStance1.inputs.ConstraintWrench.fFrontRightToe.x0 = 100 * ones(3, 1);

bounds.TrotStance1.inputs.ConstraintWrench.fBackLeftToe.lb = -5000 * ones(3, 1);
bounds.TrotStance1.inputs.ConstraintWrench.fBackLeftToe.ub = 5000 * ones(3, 1);
bounds.TrotStance1.inputs.ConstraintWrench.fBackLeftToe.x0 = 100 * ones(3, 1);

bounds.TrotStance1.params.pFrontRightToe.lb = -[10; 10; 0];
bounds.TrotStance1.params.pFrontRightToe.ub = [10; 10; 0];
bounds.TrotStance1.params.pFrontRightToe.x0 = [0.3; -0.2; 0];

bounds.TrotStance1.params.pBackLeftToe.lb = -[10; 10; 0];
bounds.TrotStance1.params.pBackLeftToe.ub = [10; 10; 0];
bounds.TrotStance1.params.pBackLeftToe.x0 = [-0.3; 0.2; 0];

bounds.TrotStance1.params.atime.lb = -(2 * pi) * ones(degree*num_hol, 1);
bounds.TrotStance1.params.atime.ub = (2 * pi) * ones(degree*num_hol, 1);
bounds.TrotStance1.params.atime.x0 = zeros(degree*num_hol, 1);

bounds.TrotStance1.params.ptime.lb = [bounds.TrotStance1.time.tf.lb, ...
  bounds.TrotStance1.time.t0.lb];
bounds.TrotStance1.params.ptime.ub = [bounds.TrotStance1.time.tf.ub, ...
  bounds.TrotStance1.time.t0.ub];
bounds.TrotStance1.params.ptime.x0 = [bounds.TrotStance1.time.tf.x0, ...
  bounds.TrotStance1.time.t0.x0];

if num_nonhol > 1
  bounds.TrotStance1.params.atimenh.lb = -(2 * pi) * ones(degree*num_nonhol, 1);
  bounds.TrotStance1.params.atimenh.ub = (2 * pi) * ones(degree*num_nonhol, 1);
  bounds.TrotStance1.params.atimenh.x0 = zeros(degree*num_nonhol, 1);

  bounds.TrotStance1.params.ptimenh.lb = [bounds.TrotStance1.time.tf.lb, ...
    bounds.TrotStance1.time.t0.lb];
  bounds.TrotStance1.params.ptimenh.ub = [bounds.TrotStance1.time.tf.ub, ...
    bounds.TrotStance1.time.t0.ub];
  bounds.TrotStance1.params.ptimenh.x0 = [bounds.TrotStance1.time.tf.x0, ...
    bounds.TrotStance1.time.t0.x0];

  bounds.TrotStance1.timenh.kp = bounds.TrotStance1.time.kp;
  bounds.TrotStance1.timenh.kd = bounds.TrotStance1.time.kd;
end

bounds.TrotStance1.time.kp = KP;
bounds.TrotStance1.time.kd = KD;


% Trot Impact One
bounds.TrotImpact1 = model_bounds;

% Trot Stance Two
bounds.TrotStance2 = model_bounds;

bounds.TrotStance2.time.t0.lb = 0;
bounds.TrotStance2.time.t0.ub = 0;
bounds.TrotStance2.time.t0.x0 = 0;

bounds.TrotStance2.time.tf.lb = 0.50;
bounds.TrotStance2.time.tf.ub = 1.20;
bounds.TrotStance2.time.tf.x0 = 0.55;

bounds.TrotStance2.time.duration.lb = 0.50;
bounds.TrotStance2.time.duration.ub = 1.20;
bounds.TrotStance2.time.duration.x0 = 0.55;

bounds.TrotStance2.states.x.x0(hip_joint_indices) = ...
  deg2rad(signRL*theta0+sign_trot2*gamma);
bounds.TrotStance2.states.x.x0(knee_joint_indices) = knee0 * ones(4, 1);

bounds.TrotStance2.inputs.Control.u.lb = [umin_hip * ones(netDOF/2, 1); ...
  umin_knee * ones(netDOF/2, 1)];
bounds.TrotStance2.inputs.Control.u.ub = [umax_hip * ones(netDOF/2, 1); ...
  umax_knee * ones(netDOF/2, 1)];
bounds.TrotStance2.inputs.Control.u.x0 = zeros(netDOF, 1);

bounds.TrotStance2.inputs.ConstraintWrench.fFrontLeftToe.lb = ...
  -5000 * ones(3, 1);
bounds.TrotStance2.inputs.ConstraintWrench.fFrontLeftToe.ub = 5000 * ones(3, 1);
bounds.TrotStance2.inputs.ConstraintWrench.fFrontLeftToe.x0 = 100 * ones(3, 1);

bounds.TrotStance2.inputs.ConstraintWrench.fBackRightToe.lb = ...
  -5000 * ones(3, 1);
bounds.TrotStance2.inputs.ConstraintWrench.fBackRightToe.ub = 5000 * ones(3, 1);
bounds.TrotStance2.inputs.ConstraintWrench.fBackRightToe.x0 = 100 * ones(3, 1);

bounds.TrotStance2.params.pFrontLeftToe.lb = -[10; 10; 0];
bounds.TrotStance2.params.pFrontLeftToe.ub = [10; 10; 0];
bounds.TrotStance2.params.pFrontLeftToe.x0 = [0.5; 0.2; 0];
%
bounds.TrotStance2.params.pBackRightToe.lb = -[10; 10; 0];
bounds.TrotStance2.params.pBackRightToe.ub = [10; 10; 0];
bounds.TrotStance2.params.pBackRightToe.x0 = [-0.1; -0.2; 0];

bounds.TrotStance2.params.atime.lb = -(2 * pi) * ones(degree*num_hol, 1);
bounds.TrotStance2.params.atime.ub = (2 * pi) * ones(degree*num_hol, 1);
bounds.TrotStance2.params.atime.x0 = zeros(degree*num_hol, 1);

bounds.TrotStance2.params.ptime.lb = [bounds.TrotStance2.time.tf.lb, ...
  bounds.TrotStance2.time.t0.lb];
bounds.TrotStance2.params.ptime.ub = [bounds.TrotStance2.time.tf.ub, ...
  bounds.TrotStance2.time.t0.ub];
bounds.TrotStance2.params.ptime.x0 = [bounds.TrotStance2.time.tf.x0, ...
  bounds.TrotStance2.time.t0.x0];

if num_nonhol > 1
  bounds.TrotStance2.params.atimenh.lb = -(2 * pi) * ones(degree*num_nonhol, 1);
  bounds.TrotStance2.params.atimenh.ub = (2 * pi) * ones(degree*num_nonhol, 1);
  bounds.TrotStance2.params.atimenh.x0 = zeros(degree*num_nonhol, 1);

  bounds.TrotStance2.params.ptimenh.lb = [bounds.TrotStance2.time.tf.lb, ...
    bounds.TrotStance2.time.t0.lb];
  bounds.TrotStance2.params.ptimenh.ub = [bounds.TrotStance2.time.tf.ub, ...
    bounds.TrotStance2.time.t0.ub];
  bounds.TrotStance2.params.ptimenh.x0 = [bounds.TrotStance2.time.tf.x0, ...
    bounds.TrotStance2.time.t0.x0];

  bounds.TrotStance2.timenh.kp = bounds.TrotStance2.time.kp;
  bounds.TrotStance2.timenh.kd = bounds.TrotStance2.time.kd;

end

bounds.TrotStance2.time.kp = KP;
bounds.TrotStance2.time.kd = KD;


% Trot Impact Two
bounds.TrotImpact2 = model_bounds;
