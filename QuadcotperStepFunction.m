function [NextObs,Reward,IsDone,LoggedSignals] = ...
    QuadcotperStepFunction(Action, LoggedSignals, environment)
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

%% Upack the inputs
% Unpack the loggedsignals for clearity
state = LoggedSignals.State;
r = state(1:3);
rdot = state(4:6);
E = state(7:9);
Edot = state(10:12);
thrust = state(13);

%Setup state vector
s = environment.s;
s(1:3) = r;
s(4:6) = rdot;
s(7:9) = E; 
s(10:12) = Edot;
s(13) = thrust;

current_control_state = LoggedSignals.Control;
cmd_roll = current_control_state(1);
cmd_pitch = current_control_state(2);
cmd_yaw = current_control_state(3);
cmd_thrust = current_control_state(4);

% Use action to get new commanded roll, pitch, yaw, and thrust
u = environment.u;
u.Roll = cmd_roll + Action(1) * environment.roll_step;
u.Pitch = cmd_pitch + Action(2) * environment.pitch_step;
u.YawRate = cmd_yaw + Action(3) * environment.yaw_step;
u.Thrust = cmd_thrust + Action(4) * environment.thrust_step;
% if u.Thrust < environment.minThrust
%     u.Thrust = environment.minThrust;
% elseif u.Thrust > environment.maxThrust
%     u.Thrust = environment.maxThrust;
% end

u = limit_controls_to_limits(u, environment.UpperLimit, environment.LowerLimit);

next_control_state = [u.Roll; u.Pitch; u.YawRate; u.Thrust];

% Scale tanh output actions to actual range
% cmd_Roll = Action(1);
% cmd_Pitch = Action(2);
% cmd_YawRate = Action(3);
% cmd_Thrust = Action(4);
% 
% original_min = -1; %tanh min output
% original_max = 1; %tanh max output
% LowerLimit = environment.ActionInfo.LowerLimit; 
% UpperLimit = environment.ActionInfo.UpperLimit;
% 
% u = environment.u;
% u.Roll = scale_to_range(cmd_Roll, original_min, original_max, LowerLimit(1), UpperLimit(1));
% u.Pitch = scale_to_range(cmd_Pitch, original_min, original_max, LowerLimit(2), UpperLimit(2));
% u.YawRate = scale_to_range(cmd_YawRate, original_min, original_max, LowerLimit(3), UpperLimit(3));
% u.YawRThrustate = scale_to_range(cmd_Thrust, original_min, original_max, LowerLimit(4), UpperLimit(4));
% 




% Propogate the environment one step 
% (uses MATLAB UAV Toolbox derivative function)
simOut = ode45(@(~,x) derivative(environment.model, x, u,environment.e),...
    [0 environment.dt], s);

s_next = simOut.y(:, end);
%u_next = [u.Roll; u.Pitch; u.YawRate; u.Thrust];

% Transform state to observation
%LoggedSignals.State = [simOut.y(:, end); u_next];
LoggedSignals.State = s_next;
LoggedSignals.Control = next_control_state;
NextObs = LoggedSignals.State;

%% Check for error
if sum(isnan(NextObs))>0
    disp('NaN error')
    disp(s_next);
    IsDone = 1;    
else
    E_next = NextObs(7:9);
    r_next = NextObs(1:3);
    IsDone = norm(r_next) > environment.DisplacementThreshold || ...
        max(abs(E_next(1:2))) > environment.AngleThreshold || ...
        abs(E_next(3)) > environment.YawThreshold;
end

% Get reward
Action_delta = max(Action)- min(Action);
%r1 = 1*(NextObs(13)/sim_end);
%r2 = 2*(1-norm(NextObs(1:3)));
%r3 = -0.01*Action_delta;
%r4 = -0.05*(norm(NextObs(4:6))); %1*(norm(NextObs(1:3))<.5);
%r5 = -0.05*(norm(NextObs(7:9)));
%r6 = -3*IsDone;
%Reward = r1 + r2 + r3 + r4 + r5 + r6;

%r1 = DisplacementThreshold - norm(NextObs(1:3));
%r2 = -50*IsDone;
%Reward = r1 + r2;

% r1 = 1 - abs(tanh(norm(NextObs(1:3))));
% r2 = 1 - abs(norm(NextObs(4:6)));
% r3 = -0.01*Action_delta;
% r4 = -10*IsDone;
% Reward = r1 + r2 + r3 + r4;

r1 = 1 - abs(tanh(norm(NextObs(1:3))));
r2 = 0.1*(environment.AngleThreshold - abs(NextObs(4)));
r3 = 0.1*(environment.AngleThreshold - abs(NextObs(5)));
r4 = 0.1*(environment.YawThreshold - abs(NextObs(6)));
r5 = -0.1*Action_delta;
r6 = -10*IsDone;
Reward = r1 + r2 + r3 + r4 + r5 + r6;

% r1 = DisplacementThreshold - norm(state(1:3));
% r2 = AngleThreshold - norm(state(7:8));
% r3 = YawThreshold - abs(state(9));
% r4 = -50 * IsDone;
% Reward = r1 + r2 + r3 + r4;


end


function scaled_output = scale_to_range(input, original_min, original_max, target_min, target_max)
    scaled_output = (((target_max - target_min)*(input - original_min))...
        /(original_max - original_min)) + target_min;
end

function u_out = limit_controls_to_limits(u_in, upper_limit_set, lower_limit_set)
    fn = fieldnames(u_in);
    u_out = u_in;
    for i = 1:length(fn)
        if u_in.(fn{i}) > upper_limit_set(i)
            u_out.(fn{i}) = upper_limit_set(i);
        elseif u_in.(fn{i}) < lower_limit_set(i)
            u_out.(fn{i}) = lower_limit_set(i);
        end
    end
end

