%Quadcopter Renforcement Learning Simulation
%Written by: Brian Wade, 3 Jan 20
%MATLAB dependencies: RL toolbox
%Parallel toolbox is "want_parallel" or "want_gpu" is set to true


%% Initalize
clear
clc
close all
start_time=tic;

rng(0)  %set random seed

%% User inputs
intial_radius = 0; %initial start radius (random direction) from target- m
initial_deviation = 3; %inital random angular velocity - deg/s

action_range = 3; %number of actions for each motor
roll_step = 0.1; %step in radians
pitch_step = 0.1; %step in radians
yaw_step = 0.1; %step in radians/sec
thrust_step = 0.01; %step in Newtons

% Thresholds and rewards
% quadcotper angle at which to fail the episode - deg converted to rad
AngleThreshold = 80 * pi/180;
YawThreshold = 170 * pi/180;
DisplacementThreshold = 3;

max_thrust_mult = 1.05; %multiple of weight of quadcopter
min_thrust_mult = 0.95; 
roll_max = 10*(pi/180); %max roll in radians
roll_min = -roll_max; %min roll in radians
pitch_max = 10*(pi/180); %max pitch in radians
pitch_min = -pitch_max; %min pitch in radians
yaw_max = 5*(pi/180); %max yaw rate in radians
yaw_min = -yaw_max; %min yaw rate in radians

want_parallel = false;
want_gpu = false;
view_networks = true;
saveFinalAgent = true;
doTraining = true;
doSim = true;
num_sims = 10;

agent_folder = 'Agents';
image_folder = 'Images';
agentName = 'AC';

learn_rate_critic = 1e-4;
grad_threshold_critic = 0.5;
L2RegFac_critic = 1e-5; %1e-5

learn_rate_actor = 1e-3;
grad_threshold_actor = 1;
%L2RegFac_actor = 1e-4; %1e-4

criticStateFC1size = 500;
criticStateFC2size = 250;

actorFC1size = 500;
actorFC2size = 250;

DiscountFactor = 0.99;
EntropyLossWeight = 0.05; %0  0.01
NumStepsToLookAhead = 64; %64  32

MaxEpisodes = 10000;
ScoreAveragingWindowLength = 30;
StopPercentMax = 0.95;
UseDeterministicExploitation = true;
Verbose = true;

%sim run time
sim_start = 0; %start time of simulation
sim_end = 3; %end time of simulation in sec
dt = 0.01; %step size in sec

%% Initialize environment
% This uses the MATLAB UAV Toolbox
model = multirotor; % initialize model
u = control(model); % initialize control input vector
s = state(model); % initalize state vector
e = environment(model); % initalize environment

environment.model = model;
environment.u = u;
environment.s = s;
environment.e = e;
environment.dt = dt;
environment.AngleThreshold = AngleThreshold;
environment.YawThreshold = YawThreshold;
environment.DisplacementThreshold = DisplacementThreshold;
environment.roll_step = roll_step;
environment.pitch_step = pitch_step;
environment.yaw_step = yaw_step;
environment.thrust_step = thrust_step;

max_T = max_thrust_mult*environment.model.Configuration.Mass*9.81;
min_T = min_thrust_mult*environment.model.Configuration.Mass*9.81;
environment.LowerLimit = [roll_min, pitch_min, yaw_min, min_T];
environment.UpperLimit = [roll_max, pitch_max, yaw_max, max_T];


%% Max Reward
max_steps = (sim_end - sim_start) / dt;
max = max_steps* (1 + 0.1*(2* environment.AngleThreshold +...
    environment.YawThreshold + 0));
StopTrainingValue = StopPercentMax * max;

%% Multicore - Start pool of workers
%Start worker pool
if want_parallel == true
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    %delete(poolobj)
    if isempty(poolobj)
        poolsize_want = round(.9*feature('numcores'));
        if poolsize_want < feature('numcores')
            poolsize = poolsize_want;
        else
            poolsize = feature('numcores')-1;
        end
        parpool('local',poolsize);
    else
        poolsize = poolobj.NumWorkers;
    end
end

%% Create agent and image folders for saved agents and images
if ~exist(agent_folder, 'dir')
    mkdir(agent_folder)
end

if ~exist(image_folder, 'dir')
    mkdir(image_folder)
end

%% Setup Renforcement Learning Action and Observation Spaces
numObs = 13;
ObservationInfo = rlNumericSpec([numObs 1]);
ObservationInfo.Name = 'Quadcopter States';
ObservationInfo.Description = ...
    'x, y, z, dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi, thrust';

ii=0;
action_space = cell(action_range);
for i = -1:2/(action_range-1):1
    ii=ii+1;
    jj=0;
    for j = -1:2/(action_range-1):1
        jj=jj+1;
        kk=0;
        for k = -1:2/(action_range-1):1
            kk=kk+1;
            ll=0;
            for l = -1:2/(action_range-1):1
                ll=ll+1;
                action_space{ii,jj,kk,ll}=[i j k l];
            end
        end
    end
end

ActionInfo = rlFiniteSetSpec(action_space);
ActionInfo.Name = 'Quadcopter Action';
num_actions = length(ActionInfo.Elements);

%define custom environment
StepHandle = @(Action,LoggedSignals) QuadcotperStepFunction(Action,...
    LoggedSignals,environment);
ResetHandle = @() QuadcopterResetFunction(intial_radius,...
    initial_deviation, environment);
env = rlFunctionEnv(ObservationInfo, ActionInfo, StepHandle, ResetHandle);

%% Critic Network
criticPath = [
    featureInputLayer(numObs,'Name','state')
    fullyConnectedLayer(criticStateFC1size,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(criticStateFC2size,'Name','CriticStateFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];

criticNetwork = layerGraph(criticPath);

if view_networks == true
    figure
    plot(criticNetwork)
    image_file = 'criticNetwork.png';
    image_save_path = fullfile(image_folder,image_file);
    saveas(gcf,image_save_path)
end


if want_gpu == true
    criticOpts = rlRepresentationOptions('LearnRate',learn_rate_critic,...
        'GradientThreshold',grad_threshold_critic,'UseDevice',"gpu",...
        'L2RegularizationFactor',L2RegFac_critic);
else
    criticOpts = rlRepresentationOptions('LearnRate',learn_rate_critic,...
    'GradientThreshold',grad_threshold_critic, ...
    'L2RegularizationFactor',L2RegFac_critic);
end

criticNetwork_obj = dlnetwork(criticNetwork);
critic = rlValueRepresentation(criticNetwork_obj, ObservationInfo,...
    'Observation', {'state'}, criticOpts);


%% Actor Network
actorPath = [
    featureInputLayer(numObs, 'Normalization','none','Name','state')
    fullyConnectedLayer(actorFC1size, 'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(actorFC2size,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(num_actions,'Name','ActorFC4')
    softmaxLayer('Name','ActorOutput')];

actorNetwork = layerGraph(actorPath);

if view_networks == true
    figure
    plot(actorNetwork)
    image_file = 'actorNetwork.png';
    image_save_path = fullfile(image_folder,image_file);
    saveas(gcf,image_save_path)
end

actorOpts = rlRepresentationOptions('LearnRate', learn_rate_actor,...
    'GradientThreshold', grad_threshold_actor);

actorNetwork_obj = dlnetwork(actorNetwork);
actor = rlStochasticActorRepresentation(actorNetwork_obj, ObservationInfo,...
    ActionInfo, 'Observation', {'state'}, actorOpts);


%% Create Agent
steps_per_episode = ceil((sim_end - sim_start)/dt);

agentOptions = rlACAgentOptions(...
    'SampleTime', dt,...
    'EntropyLossWeight', EntropyLossWeight,...
    'DiscountFactor', DiscountFactor,...
    'NumStepsToLookAhead', NumStepsToLookAhead, ...
    'UseDeterministicExploitation', UseDeterministicExploitation);

if want_parallel == true  %set up for A3C agent
    agentOptions.NumStepsToLookAhead = NumStepsToLookAhead;
else  %set up for normal AC agent
    agentOptions.NumStepsToLookAhead = steps_per_episode; %MaxEpisodes;
end

agent = rlACAgent(actor, critic, agentOptions);


%% Setup Training for Agent
%steps per episode
saved_agent_name = strcat('trained_quadcopter_', agentName, '_agent');
SaveAgentDirectory = fullfile(agent_folder,saved_agent_name);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', MaxEpisodes,...
    'MaxStepsPerEpisode', steps_per_episode,...
    'Verbose', Verbose,...
    'Plots','training-progress',...
    'StopOnError', "off",...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue', StopTrainingValue,...
    'ScoreAveragingWindowLength', ScoreAveragingWindowLength, ...
    'SaveAgentDirectory', SaveAgentDirectory); 

if want_parallel == true
    trainOpts.UseParallel = true;
    trainOpts.ParallelizationOptions.Mode = "async";
    trainOpts.ParallelizationOptions.DataToSendFromWorkers = "gradients";
    trainOpts.ParallelizationOptions.StepsUntilDataIsSent = ...
        agentOptions.NumStepsToLookAhead;
end


%% Train the Agent
if doTraining == true
    
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('Starting the training!!!!!')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
    trainingStats = train(agent,env,trainOpts);
    
    if saveFinalAgent == true
        save(trainOpts.SaveAgentDirectory,'agent')

        figure()
        plot(trainingStats.EpisodeIndex, trainingStats.EpisodeReward,...
            '--','Color',[0.3010 0.7450 0.9330])
        hold on
        plot(trainingStats.EpisodeIndex, trainingStats.AverageReward,...
            ':','Color',[0 0.4470 0.7410], 'LineWidth',2)
        plot(trainingStats.EpisodeIndex, trainingStats.EpisodeQ0,...
            '-x','Color',[0.9290 0.6940 0.1250])
        hold off
        xlabel('Episode Number')
        ylabel('Episode Reward')
        title(strcat('Reward Training History for ', agentName, ' Agent'))
        legend('Average Reward', 'Episode Reward', 'Episode Q0',...
            'location', 'northwest')

        image_file = strcat('TrainingHistory_', agentName, '.png');
        image_save_path = fullfile(image_folder,image_file);
        set(gcf,'position',[50,50,1200,400])
        saveas(gcf,image_save_path)

    end
    
    disp(' ')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('Training Complete!!!!!')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
else
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('Loading the trained agent!!!!!')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    load(trainOpts.SaveAgentDirectory,'agent')   
end    


%% Simulate and save the results
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Simulating the Quadcopter')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')

if doSim == true  %%% need to update this
    simOptions = rlSimulationOptions('MaxSteps',...
        trainOpts.MaxStepsPerEpisode);

    totalReward = zeros(num_sims,1);
    num_steps = zeros(num_sims,1);
    for i = 1:num_sims
        experience = sim(env,agent,simOptions);
        totalReward(i) = sum(experience.Reward);
        num_steps(i) = length(experience.Reward.Time);
    end

    avg_Reward = mean(totalReward);
    avg_steps = mean(num_steps);

    disp(['Average Reward for ', num2str(num_sims), ...
        ' Simulated Episode = ',num2str(avg_Reward)])
    disp(['Average Number of Steps for ', num2str(num_sims), ...
        ' Simulated Episode = ',num2str(avg_steps)])



    pos_sim = experience.Observation.QuadcopterStates.Data(1:3,:);
    x_sim = experience.Observation.QuadcopterStates.Data(1,:);
    y_sim = experience.Observation.QuadcopterStates.Data(2,:);
    z_sim = experience.Observation.QuadcopterStates.Data(3,:);

    total_pos_sim = zeros(1,size(pos_sim,2));
    for i = 1:size(pos_sim,2)
        total_pos_sim(i) = norm(pos_sim(:,i));
    end

    figure()
    subplot(1,3,1)
    plot(total_pos_sim)
    xlabel('Time (s)')
    ylabel('Absolute Value of Position Vector (m)')
    title('Magnitude of Displacement')
    subplot(1,3,2)
    plot(x_sim,y_sim)
    xlabel('X-Direction Displacement (m)')
    ylabel('Y-Direction Displacement (m)')
    title('Lateral Displacement')
    subplot(1,3,3)
    plot(z_sim)
    xlabel('Time (s)')
    ylabel('Z-Direction Displacement (m)')
    title('Vertical Displacement')

    image_file = strcat('TrainingSample_', agentName, '.png');
    image_save_path = fullfile(image_folder,image_file);
    set(gcf,'position',[50,50,1200,400])
    saveas(gcf,image_save_path)

end

%display total time to complete tasks
tElapsed = toc(start_time); 
hour=floor(tElapsed/3600);
tRemain = tElapsed - hour*3600;
min=floor(tRemain/60);
sec = tRemain - min*60;
 
disp(' ')
disp(['Time to complete: ',num2str(hour),' hours, ',num2str(min),...
    ' minutes, ',num2str(sec),' seconds'])
