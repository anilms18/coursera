function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
cost = 0;
%Add a column of 1's to x;
x = [ones(size(X,1),1) X];
h = zeros(size(x,1),num_labels);

for i=1:m
  z2 = x(i,:)*Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  
  z3 = a2*Theta2';
  hyp = sigmoid(z3);
  h(i,:) = hyp; 
  
end

for i=1:m
  sum = 0;
  for k=1:num_labels
    yk = (y(i) == k);
    hik = h(i,k);
    first = -yk*log(hik);
    second = -(1-yk)*log(1-hik);
    sum = sum+first+second;
end
cost = cost + sum;
end
J=cost/m;

sum1 = 0;
%Regularization
for j=1:size(Theta1,1)
  for k=2:size(Theta1,2)
    square = Theta1(j,k)^2;
    sum1 = sum1 + square;
  end
end


sum2 = 0;
%Regularization
for j=1:size(Theta2,1)
  for k=2:size(Theta2,2)
    square = Theta2(j,k)^2;
    sum2 = sum2 + square;
  end
end

reg = (sum1+sum2) * lambda / (2*m); 


J = J + reg;

%Gradient
d3 = zeros(num_labels,1);
del1 = zeros(size(Theta1_grad));
del2 = zeros(size(Theta2_grad));

for i=1:m
  z2 = x(i,:)*Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  
  z3 = a2*Theta2';
  hyp = sigmoid(z3);
  
  %d3
  for k=1:num_labels
    yk = (y(i) == k);
    hik = hyp(k);
    d3(k) = hik - yk;
  end
  
  %d2
  %layer two
  first = Theta2'*d3;
  d2 = first(2:end) .* sigmoidGradient(z2)'; 
  
  
  del1 = del1 + d2*x(i,1:end);
  del2 = del2 + d3*a2(1:end);
  
  
end

  Theta1_grad = del1/m;
  Theta2_grad = del2/m;


%Regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda*Theta1)/m;
Theta2_grad = Theta2_grad + (lambda*Theta2)/m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
