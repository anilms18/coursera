function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



%Compute XTheta(1xnXnxm(1Xm)
X_Theta = X*theta;

%(mX1)
sigma = sigmoid(X_Theta);

%first term
first_term = -y.*log(sigma);

second_term = (1-y).*log(1-sigma);

sum1 = sum(first_term - second_term);
J2 = sum1/m;

%Regularized...
temp = theta;
temp(1) = 0;
reg_sum = sum(temp.^2);
reg_sum = (lambda*reg_sum)/(2*m);

J = J2 + reg_sum;

if 0
sum = 0;

for i = 1:m
    h=theta'*X(i,:)';
    first=y(i)*log(sigmoid(h) );
    second=(1-y(i))*log(1-sigmoid(h));        
    sum = sum - first - second;    
end

sum_reg = 0;
for i=2:length(theta)
    sum_reg = sum_reg + theta(i)^2;
end
    reg = lambda*(sum_reg)/(2*m);
    J = sum/m + reg;
 
end


%Gradients(1Xm)
first = sigma - y;
mult = X'*first;
grad = mult/m;
grad_sum = (lambda/m).*temp;
grad = grad + grad_sum;

if 0
  sum = zeros(size(X,2));
for j=1:length(theta)
    sum0 = 0;
    for i = 1:m
        h=theta'*X(i,:)';
        first=(sigmoid(h) - y(i)) * X(i,j);                     
        sum(j) = sum(j) + first;
    end    
end

grad2(1) = sum(1)/m;    
for j=2:length(theta)
    reg1 = lambda*theta(j)/m;
    grad2(j) = sum(j)/m + reg1;    
end


end
% =============================================================

grad = grad(:);

end
