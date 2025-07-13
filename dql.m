% @author: SH Cho
clear all; close all force; format compact

pe = pyenv();
env = py.gymnasium.make('CartPole-v1', pyargs('render_mode', "rgb_array"));

nFeatures = double(env.observation_space.shape(1)); % 4
nActions = double(env.action_space.n); % 2  
disp("nFeatures: "+ nFeatures +  ", nActions: "+ nActions)
threshold = env.spec.reward_threshold;
disp("threshold: "+ threshold)

SEED = 1;
rng(SEED)
py.random.seed(int64(SEED));
py.numpy.random.seed(int64(SEED));
env.action_space.seed(int64(SEED));

nHidden = 256
net = dlnetwork;
layers = [
 featureInputLayer(nFeatures, 'Name', 'input_layer');
 fullyConnectedLayer(nHidden, 'Name', 'hidden_1'); 
 reluLayer("Name","layer_relu1");
 fullyConnectedLayer(nActions, 'Name', 'output_layer');
];
net = addLayers(net,layers);
layout = networkDataLayout([nFeatures NaN],"CB");
net = initialize(net, layout);
%plot(net)

buffer.State = {};
buffer.Action = {};
buffer.Reward = [];
buffer.NextState = {};
buffer.IsDone = [];
buffer_max_length = 1000;

function [gradients, loss] = modelGradients(net, discount_factor, buffer, nFeatures, batch_size)
    idx= randsample(size(buffer.State, 3), batch_size);
    state = buffer.State(:, :, idx);
    dlX = single(state);
    dlX = dlarray(reshape(dlX, nFeatures, batch_size), 'CB');
    next_state = buffer.NextState(:, :, idx);
    dlX_target = single(next_state);
    dlX_target = dlarray(reshape(dlX_target, nFeatures, batch_size), 'CB');
    rewards = buffer.Reward(idx);
    dones = buffer.IsDone(idx);
    actions = buffer.Action(:,:, idx);
    actions = reshape(actions, 1, batch_size);
    
    dlYPred = forward(net, dlX);
    linearIndices = sub2ind(size(dlYPred), actions, 1:length(actions)); % Convert Subscripts to Linear Indices for Matrix
    dlYPred = dlYPred(linearIndices);
    dlYPred = dlarray(dlYPred, 'CB');
    
    dlYPred_target = forward(net, dlX_target);
    dlYPred_target = max(dlYPred_target);
    dlYPred_target = rewards + (discount_factor * dlYPred_target.*double(1-dones));
    loss = huber(dlYPred, dlYPred_target, 'TransitionPoint', 1);
    gradients = dlgradient(loss, net.Learnables);
end

batch_size = 64
discount_factor = 0.99;
epsilon = 1.0; % for exploration
min_epsilon = 0.01;
epsilon_decay = 0.99;
learningRate = 0.001;
averageGrad = []; % for ADAM
averageSqGrad = []; % for ADAM

scores_array = [];
scores_array_max_length = 100;
max_step = 1000;
num_episodes = 1000;
print_interval = 10;
total_step = 0;

for episode = 1:num_episodes
    obv = env.reset(pyargs('seed', int8(SEED)));
    state = double(obv{1});
    step = 0;
    score = 0;
    while step < max_step
        step = step+1;
        total_step = total_step +1;
        temp = py.random.random();
        if temp < epsilon
            action_python = env.action_space.sample();
            action_python = int8(action_python);
            action = action_python + 1; % 0, 1 --> 1, 2
        else
            dlX = dlarray(single(state'), 'CB'); 
            dlYPred = forward(net, dlX);
            [~, action] = max(dlYPred);
            action = extractdata(action);
            action = int8(action);
            action_python = action - 1;
        end
        obs = env.step(action_python);
        next_state = double(obs{1});
        reward = double(obs{2});
        done = int8(obs{3});
        score  = score + reward;
        if done == true
            reward = -1;
        end
        if length(buffer.State) == 0
            buffer.State = state';
            buffer.Action = action;
            buffer.Reward = [buffer.Reward, reward];
            buffer.NextState = next_state';
            buffer.IsDone = [buffer.IsDone, done];
        else
            buffer.State = cat(3, buffer.State, state');
            buffer.Action = cat(3, buffer.Action, action);
            buffer.Reward = [buffer.Reward, reward];
            buffer.NextState = cat(3, buffer.NextState, next_state');
            buffer.IsDone = [buffer.IsDone, done];
        end
        if size(buffer.State, 3) > buffer_max_length % delete the 1st items.
            buffer.State(:, :, 1) = [];
            buffer.Action(:, :, 1) = [];
            buffer.Reward(1) = [];
            buffer.NextState(:, :, 1) = [];
            buffer.IsDone(1) = [];
        end
        if size(buffer.State, 3) >= batch_size
            [gradients, loss] = dlfeval(@modelGradients, net, discount_factor, buffer, nFeatures, batch_size);
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, total_step, learningRate);
        end
        if done == true
            break;
        end
        state = next_state;
    end
    scores_array = [scores_array, score];
    if length(scores_array) > scores_array_max_length
        scores_array(1) = []; % delete the 1st item.
    end
    avg_score = mean(scores_array);
    if episode == 1 || rem(episode, print_interval) == 0
        disp("Episode " + episode + " step " + step + ", eps = " + round(epsilon,4) ...
        + ", last " + length(scores_array) + " avg score = " + round(avg_score,4))
    end
    if avg_score > threshold
        disp("Solved after " + episode + " episodes. Average Score:"  + round(avg_score,4))
        break
    end
    if epsilon > min_epsilon
        epsilon = epsilon * epsilon_decay;
    else
        epsilon = min_epsilon;
    end
end
env.close();
