% visualization of data [WIP]

% (c) Hai Zhu, TU Delft, 2020, h.zhu@tudelft.nl

clear 
clc 

animation_save_name = "centralized.avi"
data_file_name = 'test_centralized_performance.mat'

%% Animation writer object
obj = VideoWriter(animation_save_name);
obj.Quality = 100;
obj.FrameRate = 20;
open(obj);

%% Load and abstract data
data = load(data_file_name);

% nSim       = size(data.goals, 1);               % N
nSim = 1000;
nDim        = 3;
nQuad       = size(data.log_quad_state_est, 3);              % n
nObs        = size(data.log_obs_state_est, 3); % m
quad_goal   = data.goals;                       % N*3*n
quad_pos    = data.quadrotor_positions;         % N*3*n
quad_size   = data.quadrotor_sizes;             % 3*n
quad_traj_past  = data.quadrotor_past_trajectories;     % N*T*3*n
quad_traj_plan  = data.quadrotor_future_trajectories;   % N*T*3*n
quad_traj_pred  = data.quadrotor_predicted_trajectories;% N*T*3*n
obs_pos     = data.obstacle_positions;          % N*3*m
obs_size    = data.obstacle_sizes;              % N*3*m


%% Dimension
ws.xDim = [-5, 5];
ws.yDim = [-5, 5];
ws.zDim = [ 0, 3];


%% Plotting setup
% visualization configuration
cfg.ifShowFigBox        =   1;
cfg.ifShowFigGrid       =   1;
cfg.ifShowQuadSize      =   1;
cfg.ifShowQuadGoal      =   1;
cfg.ifShowQuadTrajPast  =   1;
cfg.ifShowQuadTrajPlan  =   1;
cfg.ifShowQuadTrajPred  =   1;
cfg.ifShowEgoTrajPred   =   1;      % if showing ego quad prediction traj
cfg.ifShowObsSize       =   1;
cfg.ifShowLegend        =   1;

% color of all quad
cfg.color_quad = cell(1, nQuad);
if nQuad <= 7
    color_quad(1, 1:7) = {'r', 'g', 'b', 'm' , 'c', 'k', 'y'};
else
    color_quad(1, 1:7) = {'r', 'g', 'b', 'm' , 'c', 'k', 'y'};
    for iQuad = 8 : nQuad
        color_quad(1, iQuad) = {rand(1,3)};
    end
end
% color of ego quad and others
cfg.color_quad_ego      = 'r';
cfg.color_quad_other    = 'b';
% color of goal
cfg.color_goal_ego      = 'r';
cfg.color_goal_other    = 'c';
% trasparency
cfg.alpha_ego           = 1.0;
cfg.alpha_other         = 0.7;
% color of trajectory
cfg.color_traj_past     = 'b';
cfg.color_traj_plan     = 'g';
cfg.color_traj_pred     = 'r';
% color of obstacle
cfg.color_obs           = [0.5 0.5 0.5];


%% Plot perspective
idx_ego_quad            = 1;


%% Initial plot
% main figure
fig_main = figure;
axis([ws.xDim, ws.yDim, ws.zDim]);
ax_main = fig_main.CurrentAxes;
daspect(ax_main, [1 1 1]);
rotate3d(ax_main);
view(ax_main, 3);
hold(ax_main, 'on');
ax_main.Box = cfg.ifShowFigBox;
if cfg.ifShowFigGrid
    grid(ax_main, 'on');
end

legend_handles = {};
legend_text = {};

%% quad plot
fig_quad_pos  = cell(nQuad, 1);             % quad pos
fig_quad_goal = cell(nQuad, 1);             % quad goal
fig_quad_size = cell(nQuad, 1);             % quad size
fig_quad_traj_past = cell(nQuad, 1);        % quad past traj
fig_quad_traj_plan = cell(nQuad, 1);        % quad plan traj
fig_quad_traj_pred = cell(nQuad, 1);        % quad predicted traj
for iQuad = 1 : nQuad
    % quad color
    if iQuad == idx_ego_quad
        color_pos_temp = cfg.color_quad_ego;
        color_goal_temp = cfg.color_goal_ego;
        alpha_temp = cfg.alpha_ego;
    else
        color_pos_temp = cfg.color_quad_other;
        color_goal_temp = cfg.color_goal_other;
        alpha_temp = cfg.alpha_other;
    end
    % quad pos
    fig_quad_pos{iQuad} = scatter3(ax_main, quad_pos(1, 1, iQuad), ...
        quad_pos(1, 2, iQuad), quad_pos(1, 3, iQuad), 40, 'o', ...
        'MarkerFaceColor', color_pos_temp, 'MarkerEdgeColor', color_pos_temp, ...
        'MarkerFaceAlpha', alpha_temp, 'MarkerEdgeAlpha', alpha_temp);
    % quad goal
    if cfg.ifShowQuadGoal == 1
        fig_quad_goal{iQuad} = scatter3(ax_main, quad_goal(1, 1, iQuad), ...
            quad_goal(1, 2, iQuad), quad_goal(1, 3, iQuad), 50, 's', ...
            'MarkerFaceColor', color_goal_temp, 'MarkerEdgeColor', color_goal_temp, ...
            'MarkerFaceAlpha', alpha_temp, 'MarkerEdgeAlpha', alpha_temp);
    end
    % quad size
    if cfg.ifShowQuadSize == 1
        fig_quad_size{iQuad} = plot_ellipsoid_3D(ax_main, quad_pos(1, :, iQuad)', ...
        quad_size(:, iQuad), 'FaceColor', color_pos_temp, 'FaceAlpha', 0.2, ...
        'EdgeColor', color_pos_temp, 'EdgeAlpha', 0.1);
    end
    % quad past traj
    if cfg.ifShowQuadTrajPast == 1
        fig_quad_traj_past{iQuad} = plot3(ax_main, quad_traj_past(1, :, 1, iQuad), ...
            quad_traj_past(1, :, 2, iQuad), quad_traj_past(1, :, 3, iQuad), ...
            'Color', cfg.color_traj_past, 'LineStyle', '-', 'LineWidth', 1.2);
        fig_quad_traj_past{iQuad}.Color(4) = 0.6;   % transparency
    end
    % quad plan traj
    if cfg.ifShowQuadTrajPlan == 1
        fig_quad_traj_plan{iQuad} = plot3(ax_main, quad_traj_plan(1, :, 1, iQuad), ...
            quad_traj_plan(1, :, 2, iQuad), quad_traj_plan(1, :, 3, iQuad), ...
            'Color', cfg.color_traj_plan, 'LineStyle', '-', 'LineWidth', 1.5);
        fig_quad_traj_plan{iQuad}.Color(4) = 1.0;   % transparency
    end
    % quad pred traj
    if cfg.ifShowQuadTrajPred == 1
        fig_quad_traj_pred{iQuad} = plot3(ax_main, quad_traj_pred(1, :, 1, iQuad), ...
            quad_traj_pred(1, :, 2, iQuad), quad_traj_pred(1, :, 3, iQuad), ...
            'Color', cfg.color_traj_pred, 'LineStyle', '-', 'LineWidth', 1.5);
        fig_quad_traj_pred{iQuad}.Color(4) = 0.8;   % transparency
    end
end

idx_other_quad = min([1:idx_ego_quad-1 idx_ego_quad+1:nQuad]); % Just to get a handle for the legend

legend_text = "Query robot";
legend_handles = fig_quad_pos{idx_ego_quad};
% legend_text{1} = "Query robot";
% legend_handles{1} = fig_quad_pos{idx_ego_quad};

legend_text(end+1) = "Other robot";
legend_handles(end+1) = fig_quad_pos{idx_other_quad};

if cfg.ifShowEgoTrajPred
    legend_text(end+1) = "Past trajectory";
    legend_handles(end+1) = fig_quad_traj_past{idx_ego_quad};
elseif cfg.ifShowQuadTrajPast
    legend_text(end+1) = "Past trajectory";
    legend_handles(end+1) = fig_quad_traj_past{idx_other_quad};
end

if cfg.ifShowQuadTrajPlan
    legend_text(end+1) = "Future trajectory";
    legend_handles(end+1) = fig_quad_traj_plan{1};
end

if cfg.ifShowQuadTrajPred
    legend_text(end+1) = "Predicted trajectory";
    legend_handles(end+1) = fig_quad_traj_pred{1};
end

if cfg.ifShowQuadGoal
    legend_text(end+1) = "Goal";
    legend_handles(end+1) = fig_quad_goal{1};
end


%% obs plot
fig_obs_pos  = cell(nObs, 1);               % quad pos
fig_obs_size = cell(nObs, 1);               % quad size
for jObs = 1 : nObs
    % obs pos
    fig_obs_pos{jObs} = scatter3(ax_main, obs_pos(1, 1, jObs), ...
        obs_pos(1, 2, jObs), obs_pos(1, 3, jObs), 40, 'd', ...
        'MarkerFaceColor', cfg.color_obs, 'MarkerEdgeColor', cfg.color_obs, ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
    % obs size
    if cfg.ifShowObsSize == 1
        if length(size(obs_size)) == 2
            fig_obs_size{jObs} = plot_ellipsoid_3D(ax_main, obs_pos(1, :, jObs)', ...
                obs_size(:, jObs), 'FaceColor', cfg.color_obs, 'FaceAlpha', 0.6, ...
                'EdgeColor', cfg.color_obs, 'EdgeAlpha', 0.4);
        else
            fig_obs_size{jObs} = plot_ellipsoid_3D(ax_main, obs_pos(1, :, jObs)', ...
                obs_size(1, :, jObs), 'FaceColor', cfg.color_obs, 'FaceAlpha', 0.6, ...
                'EdgeColor', cfg.color_obs, 'EdgeAlpha', 0.4);
        end
    end
end

if nObs > 0
    legend_text(end+1) = "Obstacle";
    legend_handles(end+1) = fig_obs_pos{1};
end

%% Legend
if cfg.ifShowLegend
    legend(legend_handles, legend_text)
end

%% Animation
for kStep = 1 : nSim - 100
    if(mod(kStep, 10) == 0)
        fprintf('[%s] Step [%d] \n',datestr(now,'HH:MM:SS'), kStep);
    end
    %% quad
    for iQuad = 1 : nQuad
        % color
        if iQuad == idx_ego_quad
            color_pos_temp = cfg.color_quad_ego;
            color_goal_temp = cfg.color_goal_ego;
            alpha_temp = cfg.alpha_ego;
        else
            color_pos_temp = cfg.color_quad_other;
            color_goal_temp = cfg.color_goal_other;
            alpha_temp = cfg.alpha_other;
        end
        % pos
        set(fig_quad_pos{iQuad}, 'XData', quad_pos(kStep, 1, iQuad), ...
            'YData', quad_pos(kStep, 2, iQuad), ...
            'ZData', quad_pos(kStep, 3, iQuad));
        % goal
        if cfg.ifShowQuadGoal == 1
            set(fig_quad_goal{iQuad}, 'XData', quad_goal(kStep, 1, iQuad), ...
                'YData', quad_goal(kStep, 2, iQuad), ...
                'ZData', quad_goal(kStep, 3, iQuad));
        end
        % size
        if cfg.ifShowQuadSize == 1
            [X, Y, Z] = ellipsoid(quad_pos(kStep, 1, iQuad), ...
                quad_pos(kStep, 2, iQuad), quad_pos(kStep, 3, iQuad), ...
                quad_size(1, iQuad), quad_size(2, iQuad), ...
                quad_size(3, iQuad));
            set(fig_quad_size{iQuad}, 'XData', X, 'YData', Y, 'ZData', Z);
        end
        % past traj
        if cfg.ifShowQuadTrajPast == 1
            set(fig_quad_traj_past{iQuad}, 'XData', quad_traj_past(kStep, :, 1, iQuad), ...
                'YData', quad_traj_past(kStep, :, 2, iQuad), ...
                'ZData', quad_traj_past(kStep, :, 3, iQuad));
        end
        % plan traj
        if cfg.ifShowQuadTrajPlan == 1
            set(fig_quad_traj_plan{iQuad}, 'XData', quad_traj_plan(kStep, :, 1, iQuad), ...
                'YData', quad_traj_plan(kStep, :, 2, iQuad), ...
                'ZData', quad_traj_plan(kStep, :, 3, iQuad));
        end
        % pred traj
        if cfg.ifShowQuadTrajPred == 1
            set(fig_quad_traj_pred{iQuad}, 'XData', quad_traj_pred(kStep, :, 1, iQuad), ...
                'YData', quad_traj_pred(kStep, :, 2, iQuad), ...
                'ZData', quad_traj_pred(kStep, :, 3, iQuad));
        end
    end
    %% obs
    for jObs = 1 : nObs
        % pos
        set(fig_obs_pos{jObs}, 'XData', obs_pos(kStep, 1, jObs), ...
            'YData', obs_pos(kStep, 2, jObs), ...
            'ZData', obs_pos(kStep, 3, jObs));
        % size
        if cfg.ifShowObsSize == 1
            if length(size(obs_size)) == 2
                [X, Y, Z] = ellipsoid(obs_pos(kStep, 1, jObs), ...
                    obs_pos(kStep, 2, jObs), obs_pos(kStep, 3, jObs), ...
                    obs_size(1, jObs), obs_size(2, jObs), ...
                    obs_size(3, jObs));
                set(fig_obs_size{jObs}, 'XData', X, 'YData', Y, 'ZData', Z);
            else
                [X, Y, Z] = ellipsoid(obs_pos(kStep, 1, jObs), ...
                    obs_pos(kStep, 2, jObs), obs_pos(kStep, 3, jObs), ...
                    obs_size(kStep, 1, jObs), obs_size(kStep, 2, jObs), ...
                    obs_size(kStep, 3, jObs));
                set(fig_obs_size{jObs}, 'XData', X, 'YData', Y, 'ZData', Z);
            end
        end
    end
    
    drawnow limitrate
%     pause(0.005);
    f = getframe(gcf);
    writeVideo(obj, f)
end

obj.close();