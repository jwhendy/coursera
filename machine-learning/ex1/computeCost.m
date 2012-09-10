function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X is 97x2, theta is 2x1
% could transpose X into columns of examples to use theta' * X'
% easier to just do 
%%% X * theta

% result will be a 97 x 1 matrix; want to take the difference between
% each element of result and each element of y:
%%% (X*theta) .- y

% result is another 97 x 1 matrix; need each of these squared
%%% ((X*theta) .- y) .^ 2

% sum result by rows using sum(var,1)
%%% sum((((X*theta) .- y) .^ 2), 1)

% divide everything by 2*m
J = sum((((X*theta) .- y) .^ 2), 1) / (2*m);

% =========================================================================

end
