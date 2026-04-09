%% ANFIS Student Performance Predictor
clc; clear; close all;

%% 1. Generate Synthetic Training Data
% We need historical data for the neural network to "learn".
% Columns: [Attendance, Assignments, TestMarks, FinalScore]
rng(42); % Seed for consistent results
num_students = 200;
attendance   = 100 * rand(num_students, 1);
assignments  = 100 * rand(num_students, 1);
testmarks    = 100 * rand(num_students, 1);

% Create a realistic grading formula to act as our "true" historical data
% Heavily weighted towards tests and assignments.
true_score = 0.2*attendance + 0.3*assignments + 0.5*testmarks;
true_score = min(100, max(0, true_score)); % Clamp between 0-100

trainingData = [attendance, assignments, testmarks, true_score];

%% 2. Build the Initial Fuzzy Inference System (FIS)
disp('Generating initial FIS structure...');
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3; % 3 levels for each input (Poor, Avg, Good)
opt.InputMembershipFunctionType = 'gaussmf'; % Gaussian is required for neural net backpropagation

initialFIS = genfis(trainingData(:,1:3), trainingData(:,4), opt);

%% 3. Configure and Train the ANFIS
disp('Training ANFIS using Hybrid Learning (Backpropagation + LSE)...');
anfisOpt = anfisOptions();
anfisOpt.InitialFIS = initialFIS;
anfisOpt.EpochNumber = 100;     % Run for 100 training iterations
anfisOpt.DisplayErrorValues = 0; % Hide spammy console output during training

% Train the system
[trainedFIS, trainError] = anfis(trainingData, anfisOpt);
disp('Training Complete!');

%% 4. Plot Results (For your docs folder)
% Plot 1: The Learning Curve (Shows the Neural Network reducing error over time)
figure('Name', 'ANFIS Training Error', 'NumberTitle', 'off');
plot(trainError, 'LineWidth', 2, 'Color', 'b');
xlabel('Epoch (Training Iteration)'); ylabel('Root Mean Square Error (RMSE)');
title('Neural Network Learning Curve');
grid on;

% Plot 2: The Tuned Membership Functions
figure('Name', 'Learned Membership Functions', 'NumberTitle', 'off');
subplot(1,3,1); plotmf(trainedFIS, 'input', 1); title('Attendance MFs');
subplot(1,3,2); plotmf(trainedFIS, 'input', 2); title('Assignment MFs');
subplot(1,3,3); plotmf(trainedFIS, 'input', 3); title('Test Mark MFs');

%% 5. Evaluate System with New Students
disp('----------------------------------------------------');
disp('Evaluating New Students with Tuned Neuro-Fuzzy System:');
disp('----------------------------------------------------');

% Test cases: [Attendance, Assignments, Tests]
testStudents = [
    95, 90, 88;  % Should be Good
    70, 65, 60;  % Should be Average
    40, 30, 20   % Should be Poor
];

predictions = evalfis(trainedFIS, testStudents);

for i = 1:size(testStudents,1)
    score = predictions(i);
    
    % Map the numerical output to the required categorical levels
    if score >= 70
        level = 'Good';
    elseif score >= 40
        level = 'Average';
    else
        level = 'Poor';
    end
    
    fprintf('Student %d [Att: %2d, Assgn: %2d, Test: %2d] --> Score: %5.1f | Level: %s\n', ...
            i, testStudents(i,1), testStudents(i,2), testStudents(i,3), score, level);
end
