classdef codewords
    % CODEWORDS
    % A class containing static methods for constructing permutation-invariant quantum codes.
    %
    % Supports:
    %   - (b,g)-PI code: bgcode(b,g,fullHilbert,useSparse)
    %   - (b,g,m)-PI code: bgmcode(b,g,m,fullHilbert,useSparse)
    %   - Gross codes: gross_spin13(phi), gross_spin17()
    %   - Example: four_qubit()
    %
    % Conventions:
    %   - Dicke subspace vectors are length (N+1) with weight index (w+1).
    %   - Full Hilbert vectors are length 2^N in computational basis, with qubit-1 as MSB:
    %       idx = bin2dec(bitstring) + 1.
    %
    % Notes:
    %   - For full Hilbert space, useSparse=true is strongly recommended.
    %   - Building full 2^N vectors becomes infeasible quickly as N grows.

    methods (Static)

        %% ===================== (b,g) CODE =====================
        function [logical_0, logical_1] = bgcode(b, g, fullHilbert, useSparse)
            % [logical_0, logical_1] = bgcode(b,g,fullHilbert,useSparse)
            %
            % Inputs:
            %   b, g : positive integers (see https://arxiv.org/pdf/2411.13142)
            %   fullHilbert (optional, default=false):
            %       false -> Dicke subspace (N+1)
            %       true  -> full 2^N Hilbert space
            %   useSparse (optional, default=true for fullHilbert=true):
            %       Only relevant if fullHilbert=true.
            %
            % Outputs:
            %   logical_0, logical_1 : codewords (column vectors)

            if nargin < 3, fullHilbert = false; end
            if nargin < 4, useSparse   = true;  end

            N = 2*b + g; % total number of physical qubits

            % Dicke states |D(N,w)>
            D_0         = codewords.symmetric_basis_state(N, 0, fullHilbert, useSparse);
            D_2b        = codewords.symmetric_basis_state(N, 2*b, fullHilbert, useSparse);
            D_g         = codewords.symmetric_basis_state(N, g, fullHilbert, useSparse);
            D_2b_plus_g = codewords.symmetric_basis_state(N, 2*b + g, fullHilbert, useSparse);

            logical_0 = (sqrt(2*b - g) * D_0 + sqrt(2*b + g) * D_2b) / sqrt(4*b);
            logical_1 = (sqrt(2*b - g) * D_2b_plus_g + sqrt(2*b + g) * D_g) / sqrt(4*b);

            logical_0 = logical_0 / norm(logical_0);
            logical_1 = logical_1 / norm(logical_1);
        end

        %% ===================== (b,g,m) CODE =====================
        function [logical0, logical1] = bgmcode(b, g, m, fullHilbert, useSparse)
            % [logical0, logical1] = bgmcode(b,g,m,fullHilbert,useSparse)
            %
            % Inputs:
            %   b, g, m : positive integers (see eqn 8,9 of arxiv.org/pdf/2411.13142)
            %   fullHilbert (optional, default=false):
            %       false -> Dicke subspace (N+1)
            %       true  -> full 2^N Hilbert space
            %   useSparse (optional, default=true for fullHilbert=true)
            %
            % Outputs:
            %   logical0, logical1 : codewords (column vectors)

            if nargin < 4, fullHilbert = false; end
            if nargin < 5, useSparse   = true;  end

            N = 2*b*m + g;

            gamma_vec = codewords.bgm_gamma_vectors(b, g, m);
            norm_den  = 2^m * sqrt(codewords.double_factorial(2*m - 1));

            if ~fullHilbert
                % ---- Dicke subspace representation (dim = N+1) ----
                dim = N + 1;
                logical0 = zeros(dim, 1);

                for k = 0:m
                    w   = 2*b*k;     % weight
                    idx = w + 1;     % Dicke index
                    amp = sqrt(nchoosek(m,k)) * gamma_vec(k+1) / norm_den;
                    logical0(idx) = amp;
                end

                logical0 = logical0 / norm(logical0);
                logical1 = flipud(logical0); % weight w -> N-w in Dicke basis

            else
                % ---- Full 2^N representation ----
                dim = 2^N;
                if useSparse
                    logical0 = sparse(dim, 1);
                else
                    logical0 = zeros(dim, 1);
                end

                for k = 0:m
                    w   = 2*b*k;
                    amp = sqrt(nchoosek(m,k)) * gamma_vec(k+1) / norm_den;

                    Dfull = codewords.dicke_state_full(N, w, useSparse);
                    logical0 = logical0 + amp * Dfull;
                end

                logical0 = logical0 / norm(logical0);

                % logical1 = X^{\otimes N} logical0 (bitwise NOT mapping)
                logical1 = codewords.apply_global_X_full(logical0, N);
                logical1 = logical1 / norm(logical1);
            end
        end

        %% ===================== HELPERS FOR (b,g,m) =====================
        function gamma_vec = bgm_gamma_vectors(b, g, m)
            gamma_vec = zeros(m+1, 1);
            prefactor = b^(-m/2);

            for k = 0:m
                prod1 = 1;
                for i = (k+1):m
                    prod1 = prod1 * sqrt(2*i*b - g);
                end

                prod2 = 1;
                for j = (m-k+1):m
                    prod2 = prod2 * sqrt(2*j*b + g);
                end

                gamma_vec(k+1) = prefactor * prod1 * prod2;
            end
        end

        function val = double_factorial(n)
            if n <= 0
                val = 1;
                return;
            end
            val = prod(n:-2:1);
        end

        %% ===================== SYMMETRIC / DICKE BASIS STATES =====================
        function state = symmetric_basis_state(N, w, fullHilbert, useSparse)
            % state = symmetric_basis_state(N,w,fullHilbert,useSparse)
            %
            % If fullHilbert=false: returns Dicke basis vector e_{w+1} in dim N+1.
            % If fullHilbert=true : returns normalized Dicke state |D(N,w)> in dim 2^N.

            if nargin < 3, fullHilbert = false; end
            if nargin < 4, useSparse   = true;  end

            if ~fullHilbert
                state = zeros(N + 1, 1);
                idx = w + 1;
                if idx < 1 || idx > (N + 1)
                    error('Weight w out of range for Dicke subspace.');
                end
                state(idx) = 1;
            else
                state = codewords.dicke_state_full(N, w, useSparse);
            end
        end

        function psi = dicke_state_full(N, w, useSparse)
            % psi = dicke_state_full(N,w,useSparse)
            % Normalized Dicke state |D(N,w)> in computational basis (2^N x 1).
            %
            % Convention: qubit-1 is MSB, idx = bin2dec(bitstring)+1.
            %
            % If useSparse=true, returns sparse vector with nnz = nchoosek(N,w).

            if nargin < 3, useSparse = true; end
            if w < 0 || w > N
                error('Weight w must be between 0 and N.');
            end

            dim = 2^N;

            % Fast endpoints
            if w == 0
                if useSparse
                    psi = sparse(1, 1, 1, dim, 1);
                else
                    psi = zeros(dim, 1); psi(1) = 1;
                end
                return;
            elseif w == N
                if useSparse
                    psi = sparse(dim, 1, 1, dim, 1);
                else
                    psi = zeros(dim, 1); psi(dim) = 1;
                end
                return;
            end

            combos = nchoosek(1:N, w);
            nterms = size(combos, 1);

            idxs = zeros(nterms, 1);
            for r = 1:nterms
                bits = zeros(1, N);
                bits(combos(r, :)) = 1;

                % bits (MSB first) -> integer
                n0 = 0;
                for q = 1:N
                    n0 = bitshift(n0, 1) + bits(q);
                end
                idxs(r) = n0 + 1; % MATLAB 1-based
            end

            amp = 1 / sqrt(nterms);

            if useSparse
                psi = sparse(idxs, ones(nterms,1), amp, dim, 1);
            else
                psi = zeros(dim, 1);
                psi(idxs) = amp;
            end
        end

        function y = apply_global_X_full(x, N)
            % y = apply_global_X_full(x,N)
            % Applies X^{\otimes N} to state vector x in full computational basis.
            % Mapping (0-based): |n> -> |(2^N-1)-n| (bitwise NOT).

            dim = 2^N;
            if length(x) ~= dim
                error('State length mismatch for full Hilbert space.');
            end

            if issparse(x)
                [ii, ~, vv] = find(x);
                n0 = ii - 1;
                n0_not = (dim - 1) - n0;
                jj = n0_not + 1;
                y = sparse(jj, ones(size(jj)), vv, dim, 1);
            else
                y = zeros(dim, 1);
                for idx = 1:dim
                    n0 = idx - 1;
                    n0_not = (dim - 1) - n0;
                    y(n0_not + 1) = x(idx);
                end
            end
        end

        %% ===================== GROSS 13 (TOP DICKE BLOCK) =====================
        function Jx = spin_jmat_x(j)
            dim = round(2*j + 1);
            mlist = j:-1:-j;

            Jp = zeros(dim, dim);
            for col = 1:dim
                m = mlist(col);
                mp = m + 1;
                if mp <= j
                    row = round(j - mp) + 1;
                    Jp(row, col) = sqrt(j*(j+1) - m*(m+1));
                end
            end

            Jm = Jp';
            Jx = (Jp + Jm) / 2;
        end

        function [ket0, ket1] = gross_spin13(phi)
            if nargin < 1
                phi = 0.0;
            end

            N   = 13;
            j   = N/2;
            dim = N + 1;

            c1 = sqrt(910) / 56;
            c2 = -3 * sqrt(154) / 56;
            c3 = -sqrt(770) / 56;
            c4 = sqrt(70) / 56;

            c5 = sqrt(231) / 84;
            c6 = sqrt(1365) / 84;
            c7 = -sqrt(273) / 28;
            c8 = -sqrt(3003) / 84;

            function v = ket_jm(m)
                idx0 = round(j - m);
                idx  = idx0 + 1;
                v = zeros(dim, 1);
                v(idx) = 1;
            end

            b1 = ket_jm(+13/2);
            b2 = ket_jm(+5/2);
            b3 = ket_jm(-3/2);
            b4 = ket_jm(-11/2);

            state1 = c1*b1 + c2*b2 + c3*b3 + c4*b4;
            state2 = c5*b1 + c6*b2 + c7*b3 + c8*b4;

            phase = exp(1i * phi);
            a = (sqrt(13/7)) / 2;
            bb = sqrt(1 - a^2);

            ket0 = a * state1 + bb * phase * state2;
            ket0 = ket0 / norm(ket0);

            Jx = codewords.spin_jmat_x(j);
            U  = expm(-1i * pi * Jx);

            ket1 = U * ket0;
            ket1 = ket1 / norm(ket1);
        end

        %% ===================== GROSS 17 (TOP DICKE BLOCK) =====================
        function [logical0, logical1] = gross_spin17()
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

        %% ===================== EXAMPLE 4-QUBIT CODE (FULL SPACE) =====================
        function [ket0L, ket1L] = four_qubit()
            n = 4;
            dim = 2^n;

            ket0L = zeros(dim,1);
            ket1L = zeros(dim,1);

            ket0L(idx_from_bitstr('0000')) = 1/sqrt(2);
            ket0L(idx_from_bitstr('1111')) = 1/sqrt(2);

            ket1L(idx_from_bitstr('0011')) = 1/sqrt(2);
            ket1L(idx_from_bitstr('1100')) = 1/sqrt(2);

            function idx = idx_from_bitstr(bitstr)
                idx = bin2dec(bitstr) + 1;
            end
        end

    end
end
