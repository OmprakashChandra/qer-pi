classdef code
    % CODES A class containing static methods for constructing permutation-invariant quantum codes.
    % This class provides methods to generate logical codewords for (b,g)-PI and (b,g,m)-PI codes.
    %TODO: Add more code constructions here in future like i) Other kinds of PI codes like Gross codes.
    %ii) Stabilizer codes like Shor code, Steane code etc.
    methods (Static)
        function [logical_0, logical_1] = bgcode(b, g)
            %Inputs:
            %  b - Positive integer as described in https://arxiv.org/pdf/2411.13142.
            %  g - Positive integer as described in https://arxiv.org/pdf/2411.13142.
            % Note that 2b>= g+1 must hold. @gopi-btq can we say anything about the distance here?
            % Outputs:
            %   logical_0 - Logical-0 codeword of the (b,g)-PI code.
            %   logical_1 - Logical-1 codeword of the (b,g)-PI code.
            
            N = 2 * b + g; %total number of physical qubits
            
            % Define the relevant symmetric basis states
            D_0         = code.symmetric_basis_state(N, 0);         % |J, m = -J>
            D_2b        = code.symmetric_basis_state(N, 2 * b);       % |J, m = 2b - J>
            D_g         = code.symmetric_basis_state(N, g);           % |J, m = g - J>
            D_2b_plus_g = code.symmetric_basis_state(N, 2 * b + g);     % |J, m = J>
            
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
            %Note that when g, 2b-g >= 2t+1 and m>= t+1, a (b,g,m) code has distance at least 2t+1
            % and can correct t errors.
            %Outputs: 
            %logical0, logical1: Logical-0 and Logical-1 codewords of the (b,g,m)-PI code.
            
            %total number of physical qubits 
            N = 2*b*m + g;
            %dimension of the Dicke subspace
            dim = N+1; 
            %initialize logical codewords
            logical0 = zeros(dim, 1);
            %compute coefficients gamma(b,g,m,k)
            gamma_vec = code.bgm_gamma_vectors(b,g,m); 
            %coefficients in the denominator
            norm_den = 2^m* sqrt(code.double_factorial(2*m-1));

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
    

        function [logical0, logical1] = gross_spin13()
            % Gross code logicals for spin-13/2 (dim = 14)
            dim = 14;
            % basis |m> at index m+s+1 with s=13/2 ⇒ m = 13/2, 5/2, -3/2, -11/2
            basis1 = zeros(dim,1); basis1(14) = 1; % |13/2>
            basis2 = zeros(dim,1); basis2(10) = 1; % |5/2>
            basis3 = zeros(dim,1); basis3(6)  = 1; % |-3/2>
            basis4 = zeros(dim,1); basis4(2)  = 1; % |-11/2>

            c1 = sqrt(910)/56;     c2 = -3*sqrt(154)/56;
            c3 = -sqrt(770)/56;    c4 =  sqrt(70)/56;
            c5 = sqrt(231)/84;     c6 =  sqrt(1365)/84;
            c7 = -sqrt(273)/28;    c8 = -sqrt(3003)/84;

            state1 = c1*basis1 + c2*basis2 + c3*basis3 + c4*basis4;
            state2 = c5*basis1 + c6*basis2 + c7*basis3 + c8*basis4;

            phi = 0; phase = exp(1i*phi);
            a = sqrt(105)/14; b = sqrt(91)/14;

            logical0 = a*state1 + b*phase*state2;
            logical1 = b*state1 - a*conj(phase)*state2;

            % (optional) ensure unit norm
            logical0 = logical0 / norm(logical0);
            logical1 = logical1 / norm(logical1);
        end

   
        function [logical0, logical1] = gross_spin17()
            % Gross code logicals for spin-17/2 (dim = 18)
            logical0 = [ ...
                0;
                (1/25056).*((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (1/12528).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (1/4176).*(187/87)^(1/2).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (1/72).*sqrt((17/6).*(281+2*sqrt(17290)));
                0;0;0;
                (1/25056).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
            ];

            logical1 = [ ...
                (17/13824).*sqrt((35/6).*(281+2*sqrt(17290))) ...
                + (43/3207168).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
                + (187/801792).*sqrt(65/174).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2)) ...
                + (1/9621504).*sqrt(17).* ((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2)) ...
                + (1/2405376).*sqrt(1547).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (29/6912).*sqrt((17/6).*(281+2*sqrt(17290))) ...
                + (1/4810752).*sqrt(595).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
                + (11/400896).*sqrt(1547/174).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2)) ...
                + (1/534528).*sqrt(35).* ((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2)) ...
                - (1/400896).*sqrt(65).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (1/13824).*sqrt((17017/3).*(281+2*sqrt(17290))) ...
                + (1/4810752).*sqrt(12155/2).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
                + (11/89088).*(187/87)^(1/2).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2)) ...
                + (1/4810752).*sqrt(715/2).* ((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2)) ...
                + (1/1202688).*sqrt(385/2).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (-1/2304).*sqrt((1105/6).*(281+2*sqrt(17290))) ...
                + (1/4810752).*sqrt(1547).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
                + (11/400896).*sqrt(595/174).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2)) ...
                - (7/4810752).*sqrt(91).* ((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2)) ...
                + (53/1202688).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0;0;0;
                (1/1536).*sqrt((595/6).*(281+2*sqrt(17290))) ...
                + (1/9621504).*sqrt(17).*(22.*(39.*(4619-34*sqrt(17290))).^(1/2)+17.*(210.*(281+2*sqrt(17290))).^(1/2)) ...
                + (11/801792).*sqrt(1105/174).* ((290.*(4619-34*sqrt(17290))).^(1/2)+(2639.*(281+2*sqrt(17290))).^(1/2)) ...
                + (113/9621504).* ((-2).*(663.*(4619-34*sqrt(17290))).^(1/2)+9.*(3570.*(281+2*sqrt(17290))).^(1/2)) ...
                - (7/2405376).*sqrt(91).*(4.*(357.*(4619-34*sqrt(17290))).^(1/2)-3.*(6630.*(281+2*sqrt(17290))).^(1/2));
                0 ...
            ];

            logical0 = logical0 / norm(logical0);
            logical1 = logical1 / norm(logical1);
        end

        function [ket0L, ket1L] = four_qubit()
            n   = 4;
            dim = 2^n;

            ket0L = zeros(dim,1);
            ket1L = zeros(dim,1);

            % |0_L>
            ket0L(idx_from_bitstr('0000')) = 1/sqrt(2);
            ket0L(idx_from_bitstr('1111')) = 1/sqrt(2);

            % |1_L>
            ket1L(idx_from_bitstr('0011')) = 1/sqrt(2);
            ket1L(idx_from_bitstr('1100')) = 1/sqrt(2);

           

           
            function idx = idx_from_bitstr(bitstr)
                idx = bin2dec(bitstr) + 1;
            end
        end

    end

end
    