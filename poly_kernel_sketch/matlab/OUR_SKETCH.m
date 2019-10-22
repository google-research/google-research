%--------------------------------------------------------------------------
% Apply Tensorized Random Projection sketch on DATA
%--------------------------------------------------------------------------
function DATA_SKETCH = OUR_SKETCH(DATA, K, CS_COL)

[N, D] = size(DATA);

% Generate random features
DATA_SKETCH = zeros(N, CS_COL);

for i = 1 : CS_COL
    R = 2*randi([0,1], D, K)-1; % Random D-by-K +-1 matrix
    % R = randn(D, K); % Random D-by-K normal matrix, works roughly as well
    temp = DATA * R;
    DATA_SKETCH(:,i) = prod(temp, 2); % Product of each row.
end

% Scaling
DATA_SKETCH = DATA_SKETCH / sqrt(CS_COL);
end

