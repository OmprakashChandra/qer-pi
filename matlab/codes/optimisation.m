function [fid_no, Xo1, Xo2] = optimisation(zero_l, one_l, E, rho_l, enc_flag, method)
    % optimization uses semi-definite programming to look for optimum recovery operators.
    % Inputs:
    % zero_l  - Logical-0 state.
    % one_l   - Logical-1 state.
    % rho_l   - Input state in the logical subspace.
    % E       - Cell array of Kraus operators.
    % enc_flag- Optional flag (default = 1) controlling the encoding method.
    % method  - Use "direct" for optimisation over complex Hermitian matrix, or 
    %            "lifted" for optimisation over real symmetric matrices
    
    % Outputs:
    % if method == "direct":
    % [fid_no, Xo1, Xo2] where fid_no is optimised fidelity, Xo1 is the recovery operator and Xo2 is empty.
    % if method == "lifted":
    %[fid_no, Xo1, Xo2] where fid_no is optimized fidelity, % Xo1, Xo2 - SDP variables returned by CVX. Total Choi matrix is Xo1 + 1i*Xo2. 
    
    if nargin < 4 || isempty(rho_l) || (ischar(rho_l) && strcmpi(rho_l,'none'))
        rho_l = eye(2)/2;
    end

    if nargin < 5 || isempty(enc_flag)
        enc_flag = 1;
    end

    if nargin < 6 || isempty(method)
        method = "direct";
    end

    %normalise method to lowercase string
    if iscell(method)
        method = method{1}; 
    end 
    method = lower(string(method)); 

    %validation
    if ~ismembertol(trace(rho_l), 1, 1e-7)
        error("rho not normalized")
    end
    if ~ismembertol(norm(zero_l), 1, 1e-7)
        error("zero logical not normalized")
    end
    if ~ismembertol(norm(one_l), 1, 1e-7)
        error("one logical not normalized")
    end 

    switch method 
        case "direct"
            [fid_no, Xo1] = optimisation_direct(zero_l, one_l, E, rho_l, enc_flag);
            Xo2 = [];
        case "lifted"
            [fid_no, Xo1, Xo2] = optimisation_lifted(zero_l, one_l, E, rho_l, enc_flag);
        otherwise
            error('Unsupported method. Use "direct" or "lifted".');
    end
end 

function [fid_no, Xo1] = optimisation_direct(zero_l, one_l, E, rho_l, enc_flag)

    % Default arguments
    if nargin < 5
        enc_flag = 1;
    end
    if nargin < 4 || isempty(rho_l) || (ischar(rho_l) && strcmpi(rho_l, 'none'))
        rho_l = eye(2)/2;
    end

    d   = length(zero_l);     % physical (Dicke) subspace dimension
    d_l = size(rho_l, 1);     % logical subspace dimension (usually 2)
    K   = E;                  % Kraus operators

    if enc_flag
        % Encoder: maps logical to physical (d × d_l)
        zero = [1; 0];
        one  = [0; 1];
        Uc = zero_l * zero' + one_l * one';  % (d × d_l)

        % Build C matrix for fidelity objective
        C = zeros(d_l * d, d_l * d);
        for ii = 1:numel(K)
            K1 = K{ii} * Uc;                         % (d × d_l)
            temp = transpose(rho_l * K1');           % (d × d_l)
            vec_temp = reshape(temp, [], 1);         % (d_l*d × 1)
            C = C + (vec_temp * vec_temp');          % (d_l*d × d_l*d)
        end

        % Complex Hermitian SDP
        %cvx_solver mosek
        cvx_begin quiet sdp
            variable X(d_l*d, d_l*d) complex hermitian semidefinite 
            maximize(trace(X * C))
            subject to
                TrX(X, 1, [d_l, d]) == eye(d); % TP constraint
            %X' == X %hermitian
                %X>=0; 
        cvx_end

        fid_no = real(trace(X * C));
        Xo1 = X;
        %cvx_end

    else
        % No encoding: work directly in physical space
        % Build rho in physical space
        rho = (zero_l * zero_l' * rho_l(1,1) + ...
            zero_l * one_l'  * rho_l(1,2) + ...
            one_l  * zero_l' * rho_l(2,1) + ...
            one_l  * one_l'  * rho_l(2,2));
        rho = rho / trace(rho);

        % Build C matrix
        C = zeros(d^2, d^2);
        for ii = 1:numel(K)
            K1 = K{ii};
            temp = rho * K1';
            vec_temp = reshape(temp, [], 1);
            C = C + (vec_temp * vec_temp');
        end

        % Complex Hermitian SDP
        %cvx_solver mosek
        cvx_begin quiet sdp
            variable X(d^2, d^2) complex hermitian semidefinite
            maximize(real(trace(X * C)))
            subject to
                TrX(X, 1, [d, d]) == eye(d);         % TP constraint
        cvx_end

        fid_no = real(trace(X * C));
    end
end

function [fid_no, Xo1, Xo2] = optimisation_lifted(zero_l, one_l, E, rho_l, enc_flag)

    d = length(zero_l);   % dimension of logical codewords
    %n = log2(d);          % number of qubits (assumes d is a power of 2)
    d_l = size(rho_l, 1);   % dimension of the logical subspace
    
    zero = [1; 0];
    one  = [0; 1];
    rho = (zero_l*zero_l' * rho_l(1,1) + ...
        zero_l*one_l'  * rho_l(1,2) + ...
        one_l*zero_l'  * rho_l(2,1) + ...
        one_l*one_l'   * rho_l(2,2));
    rho = rho/norm(rho); 
    
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
            %X_complex = X1 + 1i*X2; 
            X_real = [X1 -X2; X2 X1];
            maximize(trace(X_real * C_real))
            subject to
                %X_complex >= 0; 
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