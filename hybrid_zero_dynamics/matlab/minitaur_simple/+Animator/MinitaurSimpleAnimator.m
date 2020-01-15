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

classdef MinitaurSimpleAnimator < Animator.AbstractAnimator
  properties

    %% Inner leg color (uppper and lower segments)
    legIUColor = 'c';
    legILColor = 'r';

    %% Outer leg color (uppper and lower segments)
    legOUColor = 'y';
    legOLColor = 'b';

    FRColor = 'r';
    FLColor = 'g';
    BRColor = 'b';
    BLColor = 'y';

    IULineWidth = 2;
    ILLineWidth = 1;
    OULineWidth = 4;
    OLLineWidth = 3;

    %% Chassis colors
    chassisColor = 'k';

    groundColor = 'g';
  end

  properties(Access = private)
    ground;

    %% chassis CoM pos
    pChassis;

    %% Front Left leg positions
    pLink_FLL_HK
    pLink_FLL_KA

    %% Front Right leg positions
    pLink_FRR_HK
    pLink_FRR_KA

    %% Back Left leg positions
    pLink_BLL_HK
    pLink_BLL_KA

    %% Back Right leg positions
    pLink_BRR_HK
    pLink_BRR_KA


    addtext1;
    addtext2;
    addtext3;
    addtext4;
    addtext5;

    starting_index;
    next_frame_time;

    H;

    q_all;
    t_all;
  end

  properties
    updateWorldPosition logical;
  end

  methods
    function obj = MinitaurSimpleAnimator(t, q, varargin)
      obj = obj@Animator.AbstractAnimator(); % Calling super constructor

      % global gif

      obj.q_all = q;
      obj.t_all = t;

      obj.startTime = t(1);
      obj.currentTime = obj.startTime;
      obj.endTime = t(end);
      obj.updateWorldPosition = false;

      if isempty(varargin)
        [terrain.Tx, terrain.Ty] = meshgrid(-3:1:3, -3:1:3);
        terrain.Tz = 0 .* terrain.Tx;
      else
        terrain = varargin{1};
      end

      Rz = @(th) [cos(th), -sin(th), 0; sin(th), cos(th), 0; 0, 0, 1];
      Ry = @(th) [cos(th), 0, sin(th); 0, 1, 0; -sin(th), 0, cos(th)];
      Rx = @(th) [1, 0, 0; 0, cos(th), -sin(th); 0, sin(th), cos(th)];

      r = obj.q_all(1:3, end) - obj.q_all(1:3, 1);
      th = obj.q_all(4:6, end) - obj.q_all(4:6, 1);
      R = Rx(th(1)) * Ry(th(2)) * Rz(th(3));
      obj.H = [R, r; 0, 0, 0, 1];

      % Initialization
      q = obj.q_all(:, 1);

      pBase = p_ChassisCenter(q);
      ln = 0.2375 * 2;
      bd = 0.1 * 2;
      ht = 0.1;

      p_MFLL = p_motor_front_leftL_joint(q);
      p_MFRR = p_motor_front_rightR_joint(q);

      p_MBLL = p_motor_back_leftL_joint(q);
      p_MBRR = p_motor_back_rightR_joint(q);


      p_KFLL = p_knee_front_leftL_joint(q);
      p_KFRR = p_knee_front_rightR_joint(q);

      p_KBLL = p_knee_back_leftL_joint(q);
      p_KBRR = p_knee_back_rightR_joint(q);


      p_TFL = p_FrontLeftToe(q);
      p_TFR = p_FrontRightToe(q);
      p_TBL = p_BackLeftToe(q);
      p_TBR = p_BackRightToe(q);

      % Define Terrain
      obj.ground = surf(terrain.Tx, terrain.Ty, terrain.Tz);
      hold on;


      % Define Base
      obj.pChassis = obj.Rectangle(ln, bd, ht, R, pBase);

      % Define Legs

      % define thighs
      obj.pLink_FLL_HK = line([p_MFLL(1), p_KFLL(1)], ...
        [p_MFLL(2), p_KFLL(2)], [p_MFLL(3), p_KFLL(3)]);
      obj.pLink_FRR_HK = line([p_MFRR(1), p_KFRR(1)], ...
        [p_MFRR(2), p_KFRR(2)], [p_MFRR(3), p_KFRR(3)]);

      obj.pLink_BLL_HK = line([p_MBLL(1), p_KBLL(1)], ...
        [p_MBLL(2), p_KBLL(2)], [p_MBLL(3), p_KBLL(3)]);
      obj.pLink_BRR_HK = line([p_MBRR(1), p_KBRR(1)], ...
        [p_MBRR(2), p_KBRR(2)], [p_MBRR(3), p_KBRR(3)]);

      % define shins
      obj.pLink_FLL_KA = line([p_KFLL(1), p_TFL(1)], ...
        [p_KFLL(2), p_TFL(2)], [p_KFLL(3), p_TFL(3)]);
      obj.pLink_FRR_KA = line([p_KFRR(1), p_TFR(1)], ...
        [p_KFRR(2), p_TFR(2)], [p_KFRR(3), p_TFR(3)]);

      obj.pLink_BLL_KA = line([p_KBLL(1), p_TBL(1)], ...
        [p_KBLL(2), p_TBL(2)], [p_KBLL(3), p_TBL(3)]);
      obj.pLink_BRR_KA = line([p_KBRR(1), p_TBR(1)], ...
        [p_KBRR(2), p_TBR(2)], [p_KBRR(3), p_TBR(3)]);

      %
      %set(obj.ground);
      set(obj.pChassis, 'LineWidth', 1.5);

      set(obj.pLink_FLL_HK, 'LineWidth', 4, 'Color', obj.FLColor);
      set(obj.pLink_FRR_HK, 'LineWidth', 4, 'Color', obj.FRColor);
      set(obj.pLink_BLL_HK, 'LineWidth', 4, 'Color', obj.BLColor);
      set(obj.pLink_BRR_HK, 'LineWidth', 4, 'Color', obj.BRColor);

      set(obj.pLink_FLL_KA, 'LineWidth', 3, 'Color', obj.FLColor);
      set(obj.pLink_FRR_KA, 'LineWidth', 3, 'Color', obj.FRColor);
      set(obj.pLink_BLL_KA, 'LineWidth', 3, 'Color', obj.BLColor);
      set(obj.pLink_BRR_KA, 'LineWidth', 3, 'Color', obj.BRColor);

      %hold off;


    end

    function Draw(obj, t, x, varargin)

      delete(obj.pChassis);
      q = x;
      pBase = p_ChassisCenter(q);
      ln = 0.2375 * 2;
      bd = 0.1 * 2;
      ht = 0.1;

      p_MFLL = p_motor_front_leftL_joint(q);
      p_MFRR = p_motor_front_rightR_joint(q);

      p_MBLL = p_motor_back_leftL_joint(q);
      p_MBRR = p_motor_back_rightR_joint(q);


      p_KFLL = p_knee_front_leftL_joint(q);
      p_KFRR = p_knee_front_rightR_joint(q);

      p_KBLL = p_knee_back_leftL_joint(q);
      p_KBRR = p_knee_back_rightR_joint(q);


      p_TFL = p_FrontLeftToe(q);
      p_TFR = p_FrontRightToe(q);
      p_TBL = p_BackLeftToe(q);
      p_TBR = p_BackRightToe(q);

      R = eul2rotm(q(4:6)');
      % Define Base
      obj.pChassis = obj.Rectangle(ln, bd, ht, R, pBase);

      % Define Legs

      % define thighs
      set(obj.pLink_FLL_HK, 'XData', [p_MFLL(1), p_KFLL(1)], 'YData', ...
        [p_MFLL(2), p_KFLL(2)], 'ZData', [p_MFLL(3), p_KFLL(3)]);
      set(obj.pLink_FRR_HK, 'XData', [p_MFRR(1), p_KFRR(1)], 'YData', ...
        [p_MFRR(2), p_KFRR(2)], 'ZData', [p_MFRR(3), p_KFRR(3)]);

      set(obj.pLink_BLL_HK, 'XData', [p_MBLL(1), p_KBLL(1)], 'YData', ...
        [p_MBLL(2), p_KBLL(2)], 'ZData', [p_MBLL(3), p_KBLL(3)]);
      set(obj.pLink_BRR_HK, 'XData', [p_MBRR(1), p_KBRR(1)], 'YData', ...
        [p_MBRR(2), p_KBRR(2)], 'ZData', [p_MBRR(3), p_KBRR(3)]);

      % define shins
      set(obj.pLink_FLL_KA, 'XData', [p_KFLL(1), p_TFL(1)], 'YData', ...
        [p_KFLL(2), p_TFL(2)], 'ZData', [p_KFLL(3), p_TFL(3)]);
      set(obj.pLink_FRR_KA, 'XData', [p_KFRR(1), p_TFR(1)], 'YData', ...
        [p_KFRR(2), p_TFR(2)], 'ZData', [p_KFRR(3), p_TFR(3)]);

      set(obj.pLink_BLL_KA, 'XData', [p_KBLL(1), p_TBL(1)], 'YData', ...
        [p_KBLL(2), p_TBL(2)], 'ZData', [p_KBLL(3), p_TBL(3)]);
      set(obj.pLink_BRR_KA, 'XData', [p_KBRR(1), p_TBR(1)], 'YData', ...
        [p_KBRR(2), p_TBR(2)], 'ZData', [p_KBRR(3), p_TBR(3)]);


      delete(obj.addtext1);
      delete(obj.addtext2);
      delete(obj.addtext3);
      x_loc = -0.45;
      y_loc = 0;
      z_loc = linspace(0.1, 0.5, 5);
      obj.addtext1 = text(x_loc+q(1), y_loc, -z_loc(1), ...
        'leg order - FR(r) FL(g) BR(b) BL(y) ');
      obj.addtext2 = text(x_loc+q(1), y_loc, -z_loc(2), ...
        ['hips  -', num2str(rad2deg(q(7:10))')]);
      obj.addtext3 = text(x_loc+q(1), y_loc, -z_loc(3), ...
        ['knee  -', num2str(q(11:14)')]);

      drawnow
      %hold off
    end

    function CubObj = Rectangle(obj, ln, bd, ht, R, center)

      % transparency value
      alph = 0.5;

      %% Create Vertices
      x = 0.5 * ln * [-1, 1, 1, -1, -1, 1, 1, -1]';
      y = 0.5 * bd * [1, 1, 1, 1, -1, -1, -1, -1]';
      z = 0.5 * ht * [-1, -1, 1, 1, 1, 1, -1, -1]';

      %% Create Faces
      facs = [1, 2, 3, 4; ...
        5, 6, 7, 8; ...
        4, 3, 6, 5; ...
        3, 2, 7, 6; ...
        2, 1, 8, 7; ...
        1, 4, 5, 8];
      %% Rotate and Translate Vertices
      verts = zeros(3, 8);
      for i = 1:8
        verts(1:3, i) = R * [x(i); y(i); z(i)] + R * [center(1); center(2); ...
          center(3)];
      end

      CubObj = patch('Faces', facs, 'Vertices', verts', 'FaceColor', ...
        obj.chassisColor, 'FaceAlpha', alph);

    end


    function x = GetData(obj, t)
      t_start = obj.t_all(1);
      t_end = obj.t_all(end);
      delta_t = t_end - t_start;

      val = 0;

      if t < t_start || t > t_end
        val = floor((t - t_start)/delta_t);
        t = t - val * delta_t;
      end

      if t < t_start
        t = t_start;
      elseif t > t_end
        t = t_end;
      end

      n = length(obj.t_all);
      x = obj.q_all(:, 1); % Default

      a = 1;
      b = n;

      while (a <= b)
        c = floor((a + b)/2);

        if t < obj.t_all(c)
          x = obj.q_all(:, c);
          b = c - 1;
        elseif t > obj.t_all(c)
          a = c + 1;
        else
          x = obj.q_all(:, c);
          break;
        end
      end

      delta_q = obj.q_all(1:6, end) - obj.q_all(1:6, 1);

      if obj.updateWorldPosition
        Rz = @(th) [cos(th), -sin(th), 0; sin(th), cos(th), 0; 0, 0, 1];
        Ry = @(th) [cos(th), 0, sin(th); 0, 1, 0; -sin(th), 0, cos(th)];
        Rx = @(th) [1, 0, 0; 0, cos(th), -sin(th); 0, sin(th), cos(th)];

        x_orig_init = obj.q_all(1:6, 1);
        x_current_init = obj.q_all(1:6, 1);
        if val > 0
          for i = 1:val
            x_end = obj.q_all(1:6, end);

            r1 = x_current_init(1:3) - x_orig_init(1:3);
            th1 = x_current_init(4:6) - x_orig_init(4:6);
            H1 = [Rx(th1(1)) * Ry(th1(2)) * Rz(th1(3)), r1; 0, 0, 0, 1];

            r2 = x_end(1:3) - x_orig_init(1:3);
            th2 = x_end(4:6) - x_orig_init(4:6);
            H2 = [Rx(th2(1)) * Ry(th2(2)) * Rz(th2(3)), r2; 0, 0, 0, 1];

            obj.H = H1 * H2;
            T = obj.H(1:3, 4);
            roll = atan2(-obj.H(2, 3), obj.H(3, 3));
            pitch = asin(obj.H(1, 3));
            yaw = atan2(-obj.H(1, 2), obj.H(1, 1));

            x_current_init = x_orig_init(1:6) + [T; roll; pitch; yaw];
          end

          x_current = x(1:6);

          r1 = x_current_init(1:3) - x_orig_init(1:3);
          th1 = x_current_init(4:6) - x_orig_init(4:6);
          H1 = [Rx(th1(1)) * Ry(th1(2)) * Rz(th1(3)), r1; 0, 0, 0, 1];

          r2 = x_current(1:3) - x_orig_init(1:3);
          th2 = x_current(4:6) - x_orig_init(4:6);
          H2 = [Rx(th2(1)) * Ry(th2(2)) * Rz(th2(3)), r2; 0, 0, 0, 1];

          obj.H = H1 * H2;
          T = obj.H(1:3, 4);
          roll = atan2(-obj.H(2, 3), obj.H(3, 3));
          pitch = asin(obj.H(1, 3));
          yaw = atan2(-obj.H(1, 2), obj.H(1, 1));

          x(1:6) = x_orig_init(1:6) + [T; roll; pitch; yaw];
        end
      end
    end

    function [center, radius, yaw] = GetCenter(obj, t, x)
      q = x;

      center = q(1:3); %[0; q(1:2)];
      radius = 0.75;
      yaw = 0;
    end
  end
end
