function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
sum = 0;
for i = 1:m
    h=theta'*X(i,:)';
    first=y(i)*log(sigmoid(h));
    second=(1-y(i))*log(1-sigmoid(h));
    sum = sum - first - second;
end
    J = sum/m;
    
sum0 = 0;
sum1 = 0;
sum2 = 0;
for i = 1:m
    h=theta'*X(i,:)';
    first=(sigmoid(h) - y(i)) * X(i,1);    
    second=(sigmoid(h) - y(i)) * X(i,2);    
    third=(sigmoid(h) - y(i)) * X(i,3);    
    sum0 = sum0 + first;
    sum1 = sum1 + second;
    sum2 = sum2 + third;
end
grad(1) = sum0/m;
grad(2) = sum1/m;
grad(3) = sum2/m;






% =============================================================

end
