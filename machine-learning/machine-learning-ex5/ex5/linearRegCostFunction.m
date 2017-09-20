function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
sum = 0;

for i = 1:m
    h=theta'*X(i,:)';
    first=(h - y(i))^2;
    sum = sum + first;
end

sum_reg = 0;
for i=2:length(theta)
    sum_reg = sum_reg + theta(i)^2;
end
    reg = lambda*(sum_reg)/(2*m);
    J = sum/(2*m) + reg;
    
sum = zeros(size(X,2));
for j=1:length(theta)
    sum0 = 0;
    for i = 1:m
        h=theta'*X(i,:)';
        first=(h - y(i)) * X(i,j);                     
        sum(j) = sum(j) + first;
    end    
end

grad(1) = sum(1)/m;    
for j=2:length(theta)
    reg1 = lambda*theta(j)/m;
    grad(j) = sum(j)/m + reg1;    
end

% =========================================================================

grad = grad(:);

end
