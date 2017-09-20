function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
s_arr = [0.01 0.03 0.1 0.3 1 3 10 30];

res=zeros(numel(c_arr),numel(s_arr));

x1 = [1 2 1]; x2 = [0 4 -1];
for c=1:numel(c_arr)
  for s=1:numel(s_arr)
    % SVM Parameters
    C = c_arr(c); sigma = s_arr(s);
    % We set the tolerance and max_passes lower here so that the code will run
    % faster. However, in practice, you will want to run the training to
    % convergence.
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    
    predictions = svmPredict(model, Xval);
    
    res(c,s) = mean(double(predictions ~= yval));
    
  end
end

maxval = min(min(res));

[c_best s_best] = find(res==maxval);

C = c_arr(c_best(1));
sigma = s_arr(s_best(1));




% =========================================================================

end
