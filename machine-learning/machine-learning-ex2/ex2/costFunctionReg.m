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
sum = 0;

for i = 1:m
    h=theta'*X(i,:)';
    first=y(i)*log(sigmoid(h));
    second=(1-y(i))*log(1-sigmoid(h));        
    sum = sum - first - second;    
end

sum_reg = 0;
for i=2:length(theta)
    sum_reg = sum_reg + theta(i)^2;
end
    reg = lambda*(sum_reg)/(2*m);
    J = sum/m + reg;
    
sum = zeros(size(X,2));
for j=1:length(theta)
    sum0 = 0;
    for i = 1:m
        h=theta'*X(i,:)';
        first=(sigmoid(h) - y(i)) * X(i,j);                     
        sum(j) = sum(j) + first;
    end    
end

grad(1) = sum(1)/m;    
for j=2:length(theta)
    reg1 = lambda*theta(j)/m;
    grad(j) = sum(j)/m + reg1;    
end


% =============================================================

end
