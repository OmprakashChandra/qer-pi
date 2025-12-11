function [fid_no, Xo1, Xo2] = optimisation(zero_l, one_l, rho_l, E, enc_flag)
    % optimization uses semi-definite programming to look for optimum recovery operators.
    % Inputs:
    % zero_l  - Logical-0 state.
    % one_l   - Logical-1 state.
    % rho_l   - Input state in the logical subspace.
    % E       - Cell array of Kraus operators.
    % enc_flag- Optional flag (default = 1) controlling the encoding method.
    
    % Outputs:
    % fid_no  - Optimized fidelity.
    % Xo1, Xo2 - SDP variables returned by CVX.
    
    if nargin == 4
        enc_flag = 1;
    end
    
    if ~ismembertol(trace(rho_l), 1, 1e-7)
        error("rho not normalized")
    end
    if ~ismembertol(norm(zero_l), 1, 1e-7)
        error("zero logical not normalized")
    end
    if ~ismembertol(norm(one_l), 1, 1e-7)
        error("one logical not normalized")
    end
    
    d = length(zero_l);   % dimension of logical codewords
    n = log2(d);          % number of qubits (assumes d is a power of 2)
    d_l = size(rho_l, 1);   % dimension of the logical subspace
    
    zero = [1; 0];
    one  = [0; 1];
    rho = (zero_l*zero_l' * rho_l(1,1) + ...
           zero_l*one_l'  * rho_l(1,2) + ...
           one_l*zero_l'  * rho_l(2,1) + ...
           one_l*one_l'   * rho_l(2,2));
    
    K = E; % Kraus operators. 
    
    if enc_flag
        % Preparing the C-matrix with encoding.
        Uc = zero_l*zero' + one_l*one';
        C = zeros(d_l*d, d_l*d);
        for ii = 1:length(K)
            K1 = K{ii} * Uc; % Encoding Kraus operator.
            temp = transpose(rho_l * (K1)');
            Vec_temp = reshape(temp, [], 1);
            C = C + (Vec_temp * Vec_temp');
        end
        
        P = real(C);
        Q = imag(C);
        C_real = [P, -Q; Q, P];
        
        cvx_begin quiet sdp
            variable X1(d_l*d, d_l*d) symmetric semidefinite 
            variable X2(d_l*d, d_l*d) symmetric semidefinite
            X_real = [X1 -X2; X2 X1];
            maximize(trace(X_real * C_real))
            subject to
                TrX(X1, 1, [d_l, d]) == eye(d);
                TrX(X2, 1, [d_l, d]) == zeros(d);
                trace(X2 * P + X1 * Q) == 0
        cvx_end
        
        fid_no = trace(X1 * P - X2 * Q);
        Xo1 = X1;
        Xo2 = X2;
    else
        % Preparing the C-matrix without encoding.
        C = zeros(d^2, d^2);
        for ii = 1:length(K)
            K1 = K{ii};
            temp = rho * K1';
            Vec_temp = reshape(temp, [], 1);
            C = C + (Vec_temp * Vec_temp');
        end
        
        dim = [d, d];
        sys = 1;
        cvx_begin quiet sdp
            variable X(d^2, d^2)
            maximize(trace(X * C))
            subject to
                TrX(X, sys, dim) == eye(d);
                X >= 0;
        cvx_end
        
        fid_no = trace(X * C);
        Xo1 = X;
        Xo2 = [];
    end
end