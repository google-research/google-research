% TensorSketch method split from http://www.itu.dk/people/ndap/TensorSketch.m 
%
% This source code is written by Ninh Pham (ndap@itu.dk) as a part of the MADAMS project 
% Feel free to re-use and re-distribute this source code for any purpose, 
% And cite our work when you re-use this code.
%
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Apply k-Level Tensoring Count Sketch on DATA
%--------------------------------------------------------------------------
function DATA_SKETCH = FFT_CountSketch_k_Naive(DATA, K, CS_COL)

[N, D]      = size(DATA);

% Generate 2 Hash functions for N points
indexHASH   = randi(CS_COL, K, D);                  % Matrix of K x D
bitHASH     = double(randi(2, K, D) - 1.5 ) * 2;    % Matrix of K x D

%----------------------
% Count Sketch for DATA
%----------------------

DATA_SKETCH = zeros(N, CS_COL);                     % Matrix of N x CS_COL
P           = zeros(K, CS_COL);                     % Matrix of K x CS_COL

% Loop all points Xi
for Xi = 1 : N
    
    temp   = DATA(Xi, :);                         % Data Xi
    P      = zeros(K, CS_COL);                    % Matrix of K x CS_COL
    
    % Sketching each element Xij of Xi
    for Xij = 1 : D

        % For each polynomials
        for Ki = 1 : K

            iHashIndex          = indexHASH(Ki, Xij);
            iHashBit            = bitHASH(Ki, Xij);
            P(Ki, iHashIndex)   = P(Ki, iHashIndex) + iHashBit * temp(Xij);

        end           

    end

    % FFT conversion
    P = fft(P, [], 2);
    
    % Component-wise product
    temp = prod(P, 1);
    
    % iFFT conversion
    DATA_SKETCH(Xi, :) = ifft(temp);
    
end

clear indexHASH bitHASH
end

