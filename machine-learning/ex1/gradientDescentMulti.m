function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

% X is (m x n) matrix, theta (n x 1), y is (m x 1)
% need sum_i=1^m (h(theta)^{(i)} - y^{(i)}) x^{(i)}_j for all theta_j

% start with h(theta) - y
% results in (m x n) * (n x 1) = (m x 1) vector of diffs
%%% (X*theta) .- y

% now need diffs * x^{(i)}_j; conveniently, will be summed into 
% X' = (n x m) * diffs = (m x 1) = 
% diffs of h(theta)^{(i)} * theta_j = (n x 1) 
%%% (X' * ((X*theta) .- y))

% lastly, need theta = theta - alpha/m * above
theta = theta - ((alpha/m) .* (X' * ((X*theta) .- y)));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
