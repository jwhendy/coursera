>
% create design matrix for training inputs
X = [1950; 1960; 1970; 1980; 1990; 2000; 2010];
X = [X (X .^ 2)]% (X .^ 3)];

% y for training samples
y = [21.38; 7.52; 18.81; 35.9; 33.05; 99.38; 114.75]; 

% create some test data
pred = [1950:10:2050]; 
pred = pred';
pred = [pred (pred .^ 2)]% (pred .^ 3)];

% create scaled X and pred
mu = zeros(1, size(X, 2)-1);
sig = zeros(1, size(X, 2)-1);
X_sc = zeros(size(X));
pred_sc = zeros(size(pred));

for i=1:size(X, 2)
  mu(1,i) = mean(X(:, i));
  sig(1,i) = std(X(:, i));
  X_sc(:, i) = (X(:, i) .- mu(1, i)) ./ sig(1, i);
  pred_sc(:, i) = (pred(:, i) .- mu(1, i)) ./ sig(1, i);
end

% tack on ones back
X = [ones(size(X, 1) ,1) X];
X_sc = [ones(size(X_sc, 1) ,1) X_sc];
pred = [ones(size(pred, 1) ,1) pred];
pred_sc = [ones(size(pred_sc, 1) ,1) pred_sc];


% calcualte thetas
theta = pinv(X' * X) * X' * y; 
theta_sc = pinv(X_sc' * X_sc) * X_sc' * y; 

% plot!
figure;
plot(X(:, 2), y);
hold on;
plot(pred(:, 2), pred * theta, '-r');
title('Unscaled fit');

figure;
plot(X(:, 2), y);
hold on;
plot(pred(:, 2), pred_sc * theta_sc, '-r');
title('Scaled fit');

