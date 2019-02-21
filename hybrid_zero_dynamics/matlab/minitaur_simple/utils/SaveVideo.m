tag = '20190217T020535';

cur = pwd;
addpath(genpath(cur));
if ispc
FROST_ROOT = addpath('C:\Users\Avinash Siravuru\Box\repos\frost-dev\');
slash = '\';
else
FROST_ROOT = addpath('/data/repos/frost-dev/');
slash = '/';
end

addpath(FROST_ROOT)
frost_addpath


tag = '20190218T203833';
videostring = '20cm_Steplen';
load(['/data/repos/google-research/hybrid_zero_dynamics/matlab/minitaur_simple/sol/solndata_',tag,'.mat']);

%% Visualize
t_log = [tspan{1, 1}, (tspan{1}(end) + tspan{3, 1})];
q_log = [states{1, 1}.x, states{3, 1}.x];
v_log = [states{1, 1}.dx, states{3, 1}.dx];


solution_path = fullfile(cur, ['sol',slash]);

make_vid.cycles = 2;
make_vid.flag = true;
if make_vid.flag
% To save gifs: 'gifs/optimalgait_Bez4_3d.gif';
video_path = strrep(solution_path,['sol',slash],['video',slash]);
if ~exist(video_path, 'dir')
mkdir(video_path);
end
make_vid.filename = [video_path, ...
'optimalgait_',videostring,'_',tag,'.gif'];
else
make_vid.filename = '';
end
% make_vid.visibility = 'off';
make_vid.pov = [35, 15];
plot_frames = false;
PlotMinitaurSimple(t_log, q_log, make_vid, plot_frames);