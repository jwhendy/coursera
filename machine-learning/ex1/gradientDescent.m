function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

% need to sum (theta_0 * x_0 + theta_1 * x_1)*x_j

% All but the x_j can be reused from the computeCost function:
%%% (X*theta) .- y

% now we have a (97x2)*(2x1) .- (97x1) = 97x1 matrix of
% differences between h(theta) and each training example y

% we want the above * X, stored in columns corresponding to theta
% transposing X and multiplying by the vector of diffs will sum
% the result as well
%%% X' * ((X*theta) .- y)

% finally, multiply by alpha/m and incremend theta
theta = theta - ((alpha/m) .* (X' * ((X*theta) .- y)));



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end


end
