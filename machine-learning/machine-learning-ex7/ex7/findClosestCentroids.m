function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
m=size(X,1);
for i=1:m
  min_dist = intmax;  
  for c=1:K
    dist = 0;
    for j=1:size(X,2)
      dist=dist+(X(i,j)-centroids(c,j))^2;
    end
    dist = sqrt(dist);
    if dist < min_dist
      min_dist=dist;
      best_c = c;
    end
  end  
  idx(i)=best_c;
end







% =============================================================

end

