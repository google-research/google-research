function h = fullscreen(h)
% -------------------------------------------------------------------------
%FULLSCREEN met resize the figure for full screen dimensions. If no figure 
% handle is given as argument, the current figure is used.
%
% EXAMPLE1: x = -20:20;
%           h(1) = figure();
% 			plot(x,(x).^2);
% 			title('f(x)=x^2'); xlabel('X'); ylabel('Y');
%           fullscreen(); % Put the figure in full screen
% EXAMPLE2: x = -20:20;
%           h(1) = figure();
% 			plot(x,(x).^2);
% 			title('f(x)=x^2'); xlabel('X'); ylabel('Y');
%           fullscreen(h); % Put the figure in full screen
% 
% -------------------------------------------------------------------------
% Inputs:
%    h - (Optional) figure handle. If no figure handle is given, the current
%        figure is used.
%
% Outputs:
%    h - (optional) return the figure handle.
% -------------------------------------------------------------------------
% Other m-files required:   none
% Subfunctions:             none
% MAT-files required:       none
% Other file required:      none
% See also:                 
%
% -------------------------------------------------------------------------
% Copyright 2014
% Author: Alexandre Willame
% February 2013; Last revision: 13-October-2014
% -------------------------------------------------------------------------
%------------- BEGIN CODE --------------
if(~exist('h','var'))
    h       = gcf;
end
oldunitsh    = get(h,'units');
set(h,'units','normalized');
set(h, 'Position', [0 0 1 1 ] ); % Resize figure to full screen
% set units back from normalized to pixels
set(h,'units',oldunitsh);
 %------------- END OF CODE --------------
 % Old code
% screen_size = get(0, 'ScreenSize');                        % Retreive size of screen
% set(h, 'Position', [0 0 screen_size(3) screen_size(4) ] ); % Resize figure to full screen
end
