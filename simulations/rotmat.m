function [cwR,ccwR] = rotmat(numDim,theta,rotPlane)
cwR = eye(numDim); ccwR = eye(numDim);
for i = 1:numDim
    if ismember(i, rotPlane)
        for j = 1:numDim
            if i ~= j && ismember(j, rotPlane)
                cwR(j,j) = cos(theta); 
                ccwR(j,j) = cos(theta);
                %cw
                cwR(i,j) = -sin(theta);
                cwR(j,i) = sin(theta);
                %ccw
                ccwR(j,i) = -sin(theta);
                ccwR(i,j) = sin(theta);

            elseif i == j
                cwR(i, i) = cos(theta);
                ccwR(i, i) = cos(theta);
            else
                cwR(i, j) = 0;
                ccwR(i, j) = 0;
            end
        end
    end
end
end