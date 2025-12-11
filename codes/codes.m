classdef codes
    methods (Static)
        function [logical_0, logical_1] = bgcode(b, g)
            %Parameters
            %%%%%%%%%%%%
            %   b - Parameter controlling the number of qubits and excitations.
            %   g - Parameter affecting code construction.
            % Outputs:
            %   logical_0 - Logical-0 codeword of the (b,g)-PI code.
            %   logical_1 - Logical-1 codeword of the (b,g)-PI code.
            
            % Total number of qubits
            N = 2 * b + g;
            
            % Define the relevant symmetric basis states
            D_0         = codes.symmetric_basis_state(N, 0);         % |J, m = -J>
            D_2b        = codes.symmetric_basis_state(N, 2 * b);       % |J, m = 2b - J>
            D_g         = codes.symmetric_basis_state(N, g);           % |J, m = g - J>
            D_2b_plus_g = codes.symmetric_basis_state(N, 2 * b + g);     % |J, m = J>
            
            % Construct the logical-0 and logical-1 codewords
            logical_0 = (sqrt(2 * b - g) * D_0 + sqrt(2 * b + g) * D_2b) / sqrt(4 * b);
            logical_1 = (sqrt(2 * b - g) * D_2b_plus_g + sqrt(2 * b + g) * D_g) / sqrt(4 * b);
            
            % Normalize the codewords
            logical_0 = logical_0 / norm(logical_0);
            logical_1 = logical_1 / norm(logical_1);
        end

        function [logical0, logical1] = bgmcode(b,g,m)
            %Parameters
            %%%%%%%%%%%%
            %b,g,m: Positive integers (see eqn 8,9 of arxiv.org/pdf/2411.13142)
            %Outputs: 
            %logical0, logical1: Logical-0 and Logical-1 codewords of the (b,g,m)-PI code.
            
            %total number of physical qubits 
            N = 2*b*m + g;
            %dimension of the Dicke subspace
            dim = N+1; 
            %initialize logical codewords
            logical0 = zeros(dim, 1);
            %compute coefficients gamma(b,g,m,k)
            gamma_vec = codes.bgm_gamma_vectors(b,g,m); 
            %coefficients in the denominator
            norm_den = 2^m* sqrt(codes.double_factorial(2*m-1));

            %fill in the amplitudes
            for k = 0:m
                w = 2*b*k; %weight of the Dicke state
                idx = w+1; 
                amp = sqrt(nchoosek(m,k))*gamma_vec(k+1)/norm_den;
                logical0(idx) = amp;
            end  

            logical0 = logical0/norm(logical0); %normalize logical-0 codeword

            %logical-1 is exactly the flipped version of logical-0. |1>_L = X^{otimes N} |0>_L
            logical1 = flipud(logical0); 
        end 

        function gamma_vec = bgm_gamma_vectors(b,g,m)

            gamma_vec = zeros(m+1,1); 
            prefactor = b^(-m/2); 
            for k = 0:m
                prod1 = 1; 
                for i = k+1:m 
                    prod1 = prod1* sqrt(2*i*b - g);
                end 
                prod2 =1; 
                for j = (m-k+1):m
                    prod2 = prod2* sqrt(2*j*b + g); 
                end 
                gamma_vec(k+1) = prefactor* prod1* prod2;
            end 
        end 

        function val = double_factorial(n)
            val = prod(n:-2:1); 
        end 

        function state = symmetric_basis_state(N, m)
            % SYMMETRIC_BASIS_STATE Computes a symmetric basis state for given parameters.
            % Inputs:
            %   N - Total number of qubits.
            %   m - Parameter defining the basis state.
            % Output:
            %   state - A (N+1) dimensional column vector with N zeros and 1 one at (m+1)th position.
            
            state = zeros(N + 1, 1);
            index = m + 1;  % MATLAB indices start at 1
            if index >= 1 && index <= (N + 1)
                state(index) = 1;
            else
                error('Index out of range for symmetric basis state.');
            end
        end
    end 
end 
    