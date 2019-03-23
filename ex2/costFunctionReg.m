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

for i = 1:m
  h_theta_x = sigmoid(theta' * X(i, :)');
  left_fac = -y(i) * log(h_theta_x);
  right_fac = (1 - y(i)) * log(1 - h_theta_x);
  J += (left_fac - right_fac);
endfor

J /= m;

theta_sum_sq = sum(theta(2:end) .^ 2);
right_comp = lambda * theta_sum_sq / (2 * m);
J += right_comp;

theta_length = length(theta);
for j = 1:theta_length
  for i = 1:m
    h_theta_x = sigmoid(theta' * X(i, :)');
    subtr = h_theta_x - y(i);
    left_comp = subtr * X(i, j);
    grad(j) += left_comp;
  endfor
  if (j > 1)
    right_comp = lambda * theta(j);
    grad(j) += right_comp;
  endif
endfor

grad ./= m;

% =============================================================

end
