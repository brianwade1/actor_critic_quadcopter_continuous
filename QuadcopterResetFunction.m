function [InitialObservation, LoggedSignal] = ...
    QuadcopterResetFunction(radius, deviation, environment)
% Reset function to place quadcopter environment into a random initial
%state defined as a hover at a given altitude, with no roll, pitch, or yaw,
% and no linear velocity; however, there is a random angular velocity
% caused by a disterbance which the RL program must overcome.

% Initial position
spherical_1 = rand(1)*2*pi;
spherical_2 = rand(1)*2*pi;

% Initial position (world frame) in meters
x0 = radius*cos(spherical_1)*sin(spherical_2);
y0 = radius*sin(spherical_1)*sin(spherical_2);
z0 = radius*cos(spherical_2);

% Initial velocity (world frame) in meters
rdot_0 = [0; 0; 0]; %initial velocity [x; y; z] in world frame - m/s
E0 = [0; 0; 0]; %initial [pitch;roll;yaw] relative to world frame -deg

%Add initial random roll, pitch, and yaw rates (body frame)
Edot_0 = ((2* deviation * rand(3,1) - deviation)*pi/180);

% Initial thrust - that required to hover in level flight
T_0 = environment.model.Configuration.Mass * environment.e.Gravity;

% Initital control inputs
pitch_0 = 0;
roll_0 = 0;
yaw_0= 0;
thrust_0 = 0;

s(1:3) = [x0; y0; z0]; 
s(4:6) = rdot_0;
s(7:9) = E0; 
s(10:12) = Edot_0;
s(13) = T_0;

control_state = [roll_0; pitch_0; yaw_0; thrust_0];

% Return initial environment state variables as logged signals.
LoggedSignal.State = s';
LoggedSignal.Control = control_state;
InitialObservation = LoggedSignal.State;
end
