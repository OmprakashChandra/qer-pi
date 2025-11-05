classdef project2
    % project2 is a class which encapsulates functions for code simulation including:
    % 1. Computation of symmetric subspace codewords and a lifted SDP recovery.
    % 2. Converting the recovery operators X into Kraus operators in Dicke subspace.
    % 3. Using GPG methods to find the optimum pulse sequence to reach a target state,
    %    which in our case is the Kraus operators in the Dicke subspace.
    % 4. Considering an errored GPG sequence and then studying the pulse sequence and 
    %    infidelities to reach the target state.
    
    methods (Static)
        function [logical_0, logical_1] = computeCodewordsInSymmetricSubspace(b, g)
            % COMPUTECODEWORDSINSYMMETRICSUBSPACE Directly computes the logical-0
            % and logical-1 codewords in the maximum angular momentum (symmetric)
            % subspace.
            %
            % Inputs:
            %   b - Parameter controlling the number of qubits and excitations.
            %   g - Parameter affecting code construction.
            %
            % Outputs:
            %   logical_0 - Logical-0 codeword in the symmetric subspace.
            %   logical_1 - Logical-1 codeword in the symmetric subspace.
            
            % Total number of qubits
            N = 2 * b + g;
            
            % Define the relevant symmetric basis states
            D_0         = project2.symmetric_basis_state(N, 0);         % |J, m = -J>
            D_2b        = project2.symmetric_basis_state(N, 2 * b);       % |J, m = 2b - J>
            D_g         = project2.symmetric_basis_state(N, g);           % |J, m = g - J>
            D_2b_plus_g = project2.symmetric_basis_state(N, 2 * b + g);     % |J, m = J>
            
            % Construct the logical-0 and logical-1 codewords
            logical_0 = (sqrt(2 * b - g) * D_0 + sqrt(2 * b + g) * D_2b) / sqrt(4 * b);
            logical_1 = (sqrt(2 * b - g) * D_2b_plus_g + sqrt(2 * b + g) * D_g) / sqrt(4 * b);
            
            % Normalize the codewords
            logical_0 = logical_0 / norm(logical_0);
            logical_1 = logical_1 / norm(logical_1);
        end
    
        function state = symmetric_basis_state(N, m)
            % SYMMETRIC_BASIS_STATE Computes a symmetric basis state for given parameters.
            % Inputs:
            %   N - Total number of qubits.
            %   m - Parameter defining the basis state.
            % Output:
            %   state - A column vector representing the symmetric basis state.
            
            state = zeros(N + 1, 1);
            index = m + 1;  % MATLAB indices start at 1
            if index >= 1 && index <= (N + 1)
                state(index) = 1;
            else
                error('Index out of range for symmetric basis state.');
            end
        end
    end 
        
    methods (Static)
        function [fid_no, Xo1, Xo2] = Optimize_fid_lifted_Gopi_version(zero_l, one_l, rho_l, E, enc_flag)
            % OPTIMIZE_FID_LIFTED_GOPI_VERSION Implements the lifted SDP recovery
            % optimization using semi-definite programming.
            % Inputs:
            %   zero_l  - Logical-0 state.
            %   one_l   - Logical-1 state.
            %   rho_l   - Input state in the logical subspace.
            %   E       - Cell array of Kraus operators.
            %   enc_flag- Optional flag (default = 1) controlling the encoding method.
            % Outputs:
            %   fid_no  - Optimized fidelity.
            %   Xo1, Xo2- SDP variables returned by CVX.
            
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
            
            % Determine the appropriate Kraus operator list K.
            if size(E{1}, 1) == 2  % Local Kraus operators (2x2)
                if n == 1
                    K = E;
                elseif n == 2
                    K = project2.kron_list(E, E);
                else
                    K = project2.kron_list(E, E);
                    for ii = 3:n
                        K = project2.kron_list(K, E);
                    end
                end
            elseif size(E{1}, 1) == d  % Kraus operators of dimension d x d
                K = E;
            else
                error('Unexpected dimension for Kraus operators.');
            end
            
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
                        project2.TrX(X1, 1, [d_l, d]) == eye(d);
                        project2.TrX(X2, 1, [d_l, d]) == zeros(d);
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
                        project2.TrX(X, sys, dim) == eye(d);
                        X >= 0;
                cvx_end
                
                fid_no = trace(X * C);
                Xo1 = X;
                Xo2 = [];
            end
        end
        
        function [Kraus] = Error_map(n, p)
            % ERROR_MAP Generates the error map with changing p values.
            % n is the number of qubits.
            % p is the probability of error.
            % Kraus is the cell array of Kraus operators.
            s = n/2;
            Esm = zeros(n+1);
            for j = -s:s-1
                Esm(j+s+1, j+s+2) = sqrt(s*(s+1)-j*(j+1));
            end
            Esp = Esm';
            Ex = (Esm+Esp)/2;
            Ey = (Esm-Esp)/(2*1i);
            Ez = (Esp*Esm-Esm*Esp)/2;
            E0 = ((1-0.5*(p*s*(s+1)))*eye(n+1));
            E1 = sqrt(p)*Ex;
            E2 = sqrt(p)*Ey;
            E3 = sqrt(p)*Ez;
            Kraus = {E0, E1, E2, E3};
        end

        function [fidel, R_cell] = run_simulation(logical0, logical1, p_list)
            % RUN_SIMULATION Generates an error map and runs the lifted SDP simulation.
            %
            % Inputs:
            %   logical0, logical1 - Logical codewords.
            %   p_list             - List of probabilities.
            %
            % Outputs:
            %   fidel  - Array of fidelity values.
            %   R_cell - Cell array of recovery operators.
            
            n = length(logical0) - 1;
            s = n / 2;
            Esm = zeros(n+1);
            d_l = 2;      % dimension of the logical subspace 
            % Generate the Jx, Jy and Jz operators in the appropriate space.
            for j = -s:s-1
                Esm(j+s+1, j+s+2) = sqrt(s*(s+1) - j*(j+1));
            end
            Esp = Esm';
            Ex = (Esm + Esp) / 2;
            Ey = (Esm - Esp) / (2*1i);
            Ez = (Esp*Esm - Esm*Esp) / 2;
            
            fidel = zeros(1, length(p_list));
            R_cell = cell(1, length(p_list));
            
            for j = 1:length(p_list)
                p = p_list(j);
                E0 = ((1 - 0.5*(p*s*(s+1)))) * eye(n+1);
                E1 = sqrt(p) * Ex;
                E2 = sqrt(p) * Ey;
                E3 = sqrt(p) * Ez;
                E = {E0, E1, E2, E3};
                [fidel(j), X1, X2] = project2.Optimize_fid_lifted_Gopi_version(logical0, logical1, eye(d_l)/d_l, E);
                R = X1 + 1i * X2;
                R_cell{j} = R;
            end
        end
    end 
        
    methods (Static)
        function C = kron_list(A, B)
            % KRON_LIST Returns a cell array containing the Kronecker products of
            % all combinations of elements from A and B.
            C = cell(length(A)*length(B), 1);
            for ii = 1:length(A)
                for jj = 1:length(B)
                    C{(ii-1)*length(B) + jj} = kron(A{ii}, B{jj});
                end
            end
        end
        
        function x = TrX(p, sys, dim)
            % TRX Computes the partial trace over specified subsystems.
            % p   : state vector or density matrix.
            % sys : subsystems to trace out.
            % dim : dimensions of the subsystems of p.
            
            n = length(dim);
            rdim = dim(end:-1:1);
            keep = 1:n;
            keep(sys) = [];
            dimtrace = prod(dim(sys));
            dimkeep = length(p)/dimtrace;
            
            if any(size(p) == 1)
                if size(p,1) == 1
                    p = p';
                end
                perm = n+1 - [keep(end:-1:1), sys];
                x = reshape(permute(reshape(p, rdim), perm), [dimkeep, dimtrace]);
                x = x*x';
            else 
                perm = n+1 - [keep(end:-1:1), keep(end:-1:1)-n, sys, sys-n];
                x = reshape(permute(reshape(p, [rdim, rdim]), perm), [dimkeep, dimkeep, dimtrace^2]);
                x = sum(x(:, :, 1:dimtrace+1:dimtrace^2), 3);
            end
        end 
    end
    
    methods (Static)
        function [reshaped_eigenvectors] = recovery(X, d_l, logical0, logical1)
            % RECOVERY Converts recovery operators X into Kraus operators in the Dicke subspace.
            zero = [1 0];
            one = [0 1];
            d = length(logical0); % dimension of logical codewords
            Encoder = logical0*zero + logical1*one; % Encoder matrix
            cutoff = 1e-4;
            
            reshaped_eigenvectors = cell(numel(X), 1);
            for i = 1:numel(X)
                current_matrix = X{i};          
                if issparse(current_matrix)
                    current_matrix = full(current_matrix);
                end
                current_matrix = double(current_matrix);
                [eigenvectors, eigen_matrix] = schur(current_matrix);
                eigen_values = diag(eigen_matrix);
                reshaped_eigenvectors{i} = cell(1, sum(eigen_values > cutoff));
                valid_indices = find(eigen_values > cutoff);    
                
                for j = 1:length(valid_indices)
                    idx = valid_indices(j);
                    reshaped_eigenvectors{i}{j} = sqrt(eigen_values(idx)) * transpose(reshape(eigenvectors(:, idx), [d, d_l]));
                    reshaped_eigenvectors{i}{j} = Encoder * reshaped_eigenvectors{i}{j}; 
                end
            end
        end
        
        function is_proj(A, tol)
            % IS_PROJ Checks whether each Kraus operator is proportional to a projector.
            if nargin < 2
                tol = 1e-4;
            end
            if iscell(A)
                for ii = 1:size(A, 1)
                    for jj = 1:size(A{ii}, 2)
                        Q = A{ii}{jj};
                        singular_values = svd(Q);
                        non_zero_singular_values = singular_values(abs(singular_values) > tol);
                        if length(non_zero_singular_values) == 1
                            fprintf("\n Recovery operator: %d, Matrix no: %d is proportional to a projector.\n", ii, jj);
                        else
                            if all(abs(non_zero_singular_values - non_zero_singular_values(1)) < tol)
                                fprintf("\n Recovery operator: %d, Matrix no: %d is proportional to a projector.\n", ii, jj);
                            else
                                fprintf("\n Recovery operator: %d, Matrix no: %d is not proportional to a projector.\n", ii, jj);
                            end
                        end
                    end
                end
            elseif ismatrix(A)
                singular_values = svd(A);
                non_zero_singular_values = singular_values(abs(singular_values) > tol);
                if length(non_zero_singular_values) == 1
                    fprintf("\nMatrix is proportional to a projector.\n");
                else
                    if all(abs(non_zero_singular_values - non_zero_singular_values(1)) < tol)
                        fprintf("\nMatrix is proportional to a projector.\n");
                    else
                        fprintf("\nMatrix is not proportional to a projector.\n");
                    end
                end
            end
        end
          
        function val = valueLinGPG(x, M, ESx, Phi)
            % VALUELINGPG Computes the infidelity for the linear GPG pulse sequence.
            rge = -M/2:M/2;
            Psi = zeros(M+1, 1);
            Psi(1) = 1; % all zero state
            Psi = ESx * diag(exp((-1i*pi/2)*rge)) * ESx' * Psi;  % -pi/2 rotation along Y
            Psi = diag(exp(1i*(pi/2)*rge.^2)) * Psi;              % first GPG pulse (Jz^2 term)
            Psi = ESx * diag(exp((1i*(pi/2))*rge)) * ESx' * Psi;   % pi/2 rotation along Y
            for n = 1:(M+1)
                Psi = diag(exp(1i*x(2*n-1)*rge.^2)) * Psi;      % GPG pulse
                Psi = ESx * diag(exp(1i*x(2*n)*rge)) * ESx' * Psi;  % Chi rotation along Y
            end
            val = 1 - abs(dot(Phi, Psi))^2; % infidelity
        end 
        
        function val = valueNonLinGPG(x, M, ESx, Phi)
            % VALUENONLINEGPG Computes the infidelity for the non-linear GPG pulse sequence.
            rge = -M/2:M/2;
            Psi = zeros(M+1, 1); 
            Psi(1) = 1; % all zero state
            Psi = ESx * diag(exp((-1i*pi/2)*rge)) * ESx' * Psi; % -pi/2 rotation along Y
            Psi = diag(exp(1i*(pi/2)*sin((pi/2)*rge))) * Psi;    % first non-linear GPG pulse
            Psi = ESx * diag(exp((1i*(pi/2))*rge)) * ESx' * Psi;   % pi/2 rotation along Y
            for n = 1:M
                Psi = diag(exp(1i*x(n*4-3)*sin(x(n*4-2)*rge + x(n*4-1)))) * Psi;  % non-linear GPG pulse
                Psi = ESx * diag(exp((1i*x(n*4))*rge)) * ESx' * Psi;                 % Chi rotation along Y
            end
            val = 1 - abs(dot(Phi, Psi));
        end 

        function [infidelity, unitary, x] = state_synthesis(Phi, gpg_type, tol)
            % STATE_SYNTHESIS Finds the optimum GPG pulse sequence to synthesize a target state.
            if nargin < 3
                tol = 1e-10;
            end 
            M = length(Phi) - 1;
            S = M/2;
            ESx = zeros(M+1, M+1);
            for j = -M/2:M/2-1
                ESx(j+M/2+1, j+M/2+2) = sqrt(S*(S+1) - j*(j+1));
                ESx(j+M/2+2, j+M/2+1) = sqrt(S*(S+1) - j*(j+1));
            end
            ESx = expm(ESx * 1i * pi/4);
            fv_it = 1;
            options = optimset('Display','final','TolX', tol, 'TolFun', tol, 'MaxIter', 1000000, 'MaxFunEvals', 800000);
                
            for ii = 1:50
                fv = 1;
                for cnt = 1:20
                    x = randn(2*(M+2)-2, 1);
                    if strcmp(gpg_type, 'linear')
                        [x, fval] = fminsearch(@(x) project2.valueLinGPG(x, M, ESx, Phi), x, options);
                    elseif strcmp(gpg_type, 'non-linear')
                        [x, fval] = fminsearch(@(x) project2.valueNonLinGPG(x, M, ESx, Phi), x, options);
                    else
                        error('Invalid GPG type');
                    end 
                    if fv > fval
                        fv = fval;
                        xv = x;
                    end
                end 
                if strcmp(gpg_type, 'linear')
                    [x, fval] = fminsearch(@(x) project2.valueLinGPG(x, M, ESx, Phi), xv, options);
                elseif strcmp(gpg_type, 'non-linear')
                    [x, fval] = fminsearch(@(x) project2.valueNonLinGPG(x, M, ESx, Phi), xv, options);
                else
                    error('Invalid GPG type');    
                end    
                if fv_it > fval
                    fv_it = fval;
                    xv_it = x;
                end

                if fv_it < tol
                    break;
                end  
            end
            infidelity = fv_it;
            x = xv_it;
            rge = -M/2:M/2;
            PsiOut = eye(length(Phi)); % identity matrix in the Dicke space
            PsiOut = ESx * diag(exp((-1i*pi/2)*rge)) * ESx' * PsiOut;
            PsiOut = diag(exp(1i*(pi/2)*rge.^2)) * PsiOut;
            PsiOut = ESx * diag(exp((1i*(pi/2))*rge)) * ESx' * PsiOut;
            for n = 1:(M+1)
                PsiOut = diag(exp(1i*x(2*n-1)*rge.^2)) * PsiOut;
                PsiOut = ESx * diag(exp(1i*x(2*n)*rge)) * ESx' * PsiOut;
            end
            unitary = PsiOut;
        end
        

        function [infidelity_cell, unitary_cell] = projectors(A, tol)
            % PROJECTORS Finds eigenvectors corresponding to nonzero eigenvalues of a
            % 2-D cell array of Kraus operators and uses GPG pulses to obtain recovery maps.
            infidelity_cell = cell(size(A));
            unitary_cell = cell(size(A));
            if nargin < 2
                tol = 1e-5;
            end

            if iscell(A)
                for ii = 1:size(A, 1)
                    for jj = 1:size(A{ii}, 2)
                        Q = A{ii}{jj};
                        [V, D] = eig(Q);
                        D = diag(D);
                        valid_indices = find(abs(D) > tol);
                        if isempty(valid_indices)
                            fprintf("\n • For matrix(%d,%d), all eigenvalues are below tolerance.\n", ii, jj);
                            continue;
                        else
                            fprintf("\n Matrix(%d,%d):\n", ii, jj);
                            for kk = 1:length(valid_indices)
                                idx = valid_indices(kk);
                                [infidelity, unitary, ~] = project2.state_synthesis(V(:, idx), 'linear');
                                infidelity_cell{ii}{jj}{kk} = infidelity;
                                unitary_cell{ii}{jj}{kk} = unitary;
                                
                                % Example: using the Dicke state |D^N_0>
                                Dicke = project2.symmetric_basis_state(length(V(:, idx)) - 1, 0);
                                approx_state = unitary * Dicke;
                            end
                        end 
                    end 
                    % Note: The following lines reference variables (b, g, p) that must be defined.
                    % [logical0, logical1] = project2.computeCodewordsInSymmetricSubspace(b, g);
                    % rho = (logical0*logical0' + logical1*logical1') / 2;
                    % error_map = project2.Error_map(length(logical0)-1, p);
                    % rho_e = error_map{1}*rho*error_map{1}' + ...
                    %         error_map{2}*rho*error_map{2}' + ...
                    %         error_map{3}*rho*error_map{3}' + ...
                    %         error_map{4}*rho*error_map{4}';
                    % for ll = 1:size(unitary_cell{ii}, 2)
                    %     for mm = 1:size(unitary_cell{ii}{ll}, 2)
                    %         r = (unitary_cell{ii}{ll}{mm} * Dicke) * (unitary_cell{ii}{ll}{mm} * Dicke)';
                    %         rho_r = r * rho_e * r';
                    %     end 
                    % end
                end 
            elseif ismatrix(A)
                [V, D] = eig(A);
                D = diag(D);
                valid_indices = find(abs(D) > tol);
                for ii = 1:length(valid_indices)
                    idx = valid_indices(ii);
                    disp(V(:, idx));
                    disp(D(idx));
                end
            end
        end  
    end
end
