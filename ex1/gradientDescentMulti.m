function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_length = length(theta);
temp_theta = theta

function J = computeCostDerivative(X, y, theta, j)
  m = length(y);
  J = 0;
  for i = 1:m
    h_theta_x = theta' * X(i,:)';
    inner_diff = h_theta_x - y(i);
    inner_diff_x = inner_diff * X(i, j);
    J += inner_diff_x;
  endfor
  J /= m;
endfunction

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

  for theta_iter = 1:theta_length
    computedDerivative = computeCostDerivative(X, y, theta, theta_iter);
    temp_theta(theta_iter) -= alpha * computedDerivative;
  endfor
  theta = temp_theta;









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
