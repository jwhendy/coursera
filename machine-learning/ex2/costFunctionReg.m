function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

pred = sigmoid(X*theta);

cost = 0;

for i = 1:size(pred, 1)

  cost = cost + (-y(i, 1) * log(pred(i, 1)) - ((1 - y(i, 1)) * log(1 - pred(i , 1))));

end

reg = zeros(size(theta));

for j = 2:size(theta, 1)

  cost = cost + ((lambda/2) * theta(j, 1)^2);
  reg(j, 1) = lambda * theta(j, 1);

end

J = cost/m;

grad = ((X' * (pred .- y)) .+ reg) ./ m;





% =============================================================

end
