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

classdef MinitaurSimple < RobotLinks
  % Minitaur class for FROST
  properties
    ContactPoints = struct;
    OtherPoints = struct;
    Dims = struct;
  end

  methods

    function obj = MinitaurSimple(urdf)

      % Floating base model
      base = get_base_dofs('floating');

      % Set base DOF limits
      limits = [base.Limit];

      % Simulate pinned situation

      % Set base DOF limits (Joint limits are read from URDF)
      [limits(1:6).lower] = deal(-1, -1, 0.1, deg2rad(-5), deg2rad(-7.5), ...
      deg2rad(-3));
      [limits(1:6).upper] = deal(1, 1, 0.6, deg2rad(5), deg2rad(7.5), ...
      deg2rad(3));
      [limits(1:6).velocity] = deal(100);
      [limits.effort] = deal(0);
      for i = 1:length(base)
        base(i).Limit = limits(i);
      end

      % load model from the URDF file
      obj = obj@RobotLinks(urdf, base, [], 'removeFixedJoints', true);

      %% NOTE: Both the holonomic and contact contraints are specified below


      z1 = 0.12;
      z2l = 0.24;
      z2s = 0.216;

      % Define contact frames
      footFL_frame = obj.Joints(getJointIndices(obj, 'knee_front_leftL_joint'));
      obj.ContactPoints.FrontLeftToe = CoordinateFrame( ...
        'Name', 'FrontLeftToe', ...
        'Reference', footFL_frame, ...
        'Offset', [0, 0, z2l], ...
        'R', [0, 0, 0] ...
        );

      footBL_frame = obj.Joints(getJointIndices(obj, 'knee_back_leftL_joint'));
      obj.ContactPoints.BackLeftToe = CoordinateFrame( ...
        'Name', 'BackLeftToe', ...
        'Reference', footBL_frame, ...
        'Offset', [0, 0, z2l], ...
        'R', [0, 0, 0] ...
        );

      footFR_frame = obj.Joints(getJointIndices(obj, 'knee_front_rightR_joint'));
      obj.ContactPoints.FrontRightToe = CoordinateFrame( ...
        'Name', 'FrontRightToe', ...
        'Reference', footFR_frame, ...
        'Offset', [0, 0, z2l], ...
        'R', [0, 0, 0] ...
        );

      footBR_frame = obj.Joints(getJointIndices(obj, 'knee_back_rightR_joint'));
      obj.ContactPoints.BackRightToe = CoordinateFrame( ...
        'Name', 'BackRightToe', ...
        'Reference', footBR_frame, ...
        'Offset', [0, 0, z2l], ...
        'R', [0, 0, 0] ...
        );

      % Define other frames
      chassis_frame = obj.Joints(getJointIndices(obj, 'BaseRotX'));
      obj.OtherPoints.ChassisCenter = CoordinateFrame( ...
        'Name', 'ChassisCenter', ...
        'Reference', chassis_frame, ...
        'Offset', [0, 0, 0], ...
        'R', [0, 0, 0] ...
        );


    end %Minitaur


    function ExportKinematics(obj, export_path)
      % Generates code for forward kinematics

      if ~exist(export_path, 'dir')
        mkdir(export_path);
        addpath(export_path);
      end

      % Compute positions of all joints
      for i = 1:length(obj.Joints)
        position = obj.Joints(i).computeCartesianPosition;
        orientation = obj.getEulerAngles(obj.Joints(i));
        vars = obj.States.x;
        filename = [export_path, 'p_', obj.Joints(i).Name];
        export(position, 'Vars', vars, 'File', filename);
        filename = [export_path, 'r_', obj.Joints(i).Name];
        export(orientation, 'Vars', vars, 'File', filename);
      end

      % Compute positions of contact points
      cp_fields = fields(obj.ContactPoints);
      for i = 1:length(cp_fields)
        position = obj.ContactPoints.(cp_fields{i}).computeCartesianPosition;
        orientation = obj.getEulerAngles(obj.ContactPoints.(cp_fields{i}));
        vars = obj.States.x;
        filename = [export_path, 'p_', obj.ContactPoints.(cp_fields{i}).Name];
        export(position, 'Vars', vars, 'File', filename);
        filename = [export_path, 'r_', obj.ContactPoints.(cp_fields{i}).Name];
        export(orientation, 'Vars', vars, 'File', filename);
      end

      % Compute positions of other points
      op_fields = fields(obj.OtherPoints);
      for i = 1:length(op_fields)
        position = obj.OtherPoints.(op_fields{i}).computeCartesianPosition;
        orientation = obj.getEulerAngles(obj.OtherPoints.(op_fields{i}));
        vars = obj.States.x;
        filename = [export_path, 'p_', obj.OtherPoints.(op_fields{i}).Name];
        export(position, 'Vars', vars, 'File', filename);
        filename = [export_path, 'r_', obj.OtherPoints.(op_fields{i}).Name];
        export(orientation, 'Vars', vars, 'File', filename);
      end
    end % ExportKinematics
    %

    function getRobotDimensions(obj)
      % Boxed Approximation of the Robot
      obj.Dims.chassis = [0.3, 0.13, .087];
      obj.Dims.motor = [0.068, 0.032, 0.050];
      obj.Dims.upperleg = [0.039, 0.008, 0.129];
      obj.Dims.lowerleginner = [0.017, 0.009, 0.216];
      obj.Dims.lowerlegouter = [0.017, 0.009, 0.240];

      % offsets
      obj.Dims.motor_zoffset = obj.Joints(10).Offset(3);

      obj.Dims.motor_yoffset_inner = 0.054 + 0.0275;

      obj.Dims.mFR_offset_inner = [0.2375, obj.Dims.motor_yoffset_inner, ...
        obj.Dims.motor_zoffset];
      obj.Dims.mBR_offset_inner = [-0.2375, obj.Dims.motor_yoffset_inner, ...
        obj.Dims.motor_zoffset];
      obj.Dims.mBL_offset_inner = [0.2375, -obj.Dims.motor_yoffset_inner, ...
        obj.Dims.motor_zoffset];
      obj.Dims.mFL_offset_inner = [-0.2375, -obj.Dims.motor_yoffset_inner, ...
        obj.Dims.motor_zoffset];

      obj.Dims.mFR_offset_outer = [0.2375, 0.154, obj.Dims.motor_zoffset];
      obj.Dims.mBR_offset_outer = [-0.2375, 0.154, obj.Dims.motor_zoffset];
      obj.Dims.mBL_offset_outer = [0.2375, -0.154, obj.Dims.motor_zoffset];
      obj.Dims.mFL_offset_outer = [-0.2375, -0.154, obj.Dims.motor_zoffset];

      obj.Dims.base_dof = 6;
      obj.Dims.motor_dof = 8;
      obj.Dims.knee_dof = 8;
      obj.Dims.ank_dof = 4;
      obj.Dims.total_dof = obj.Dims.base_dof + ...
        obj.Dims.motor_dof + ...
        obj.Dims.knee_dof;
      %+ ...  obj.Dims.ank_dof;

      obj.Dims.base_idx = linspace(1, obj.Dims.base_dof, obj.Dims.base_dof);
      obj.Dims.motor_idx = obj.Dims.base_idx(end) + ...
        linspace(1, obj.Dims.motor_dof, obj.Dims.motor_dof);
      obj.Dims.knee_idx = obj.Dims.motor_idx(end) + ...
        linspace(1, obj.Dims.knee_dof, obj.Dims.knee_dof);
      obj.Dims.ank_idx = obj.Dims.knee_idx(end) + ...
        linspace(1, obj.Dims.ank_dof, obj.Dims.ank_dof);


      obj.Dims.sign_left = [-1, -1, -1, -1];
      obj.Dims.sign_right = [1, 1, 1, 1];
      obj.Dims.sign_trot = [1, 1, -1, -1, 1, 1, -1, -1];
      obj.Dims.sign_IO = [1, -1, 1, -1, -1, 1, -1, 1];

    end


    function [Ax] = skew(~, v)
      % Convert from vector to skew symmetric matrix
      Ax = [0, -v(3), v(2); ...
        v(3), 0, -v(1); ...
        -v(2), v(1), 0];
    end

  end %methods

end % classdef
