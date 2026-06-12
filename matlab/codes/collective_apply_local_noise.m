function rho_out = collective_apply_local_noise(rho_c, model, param)
%COLLECTIVE_APPLY_LOCAL_NOISE  Apply identical local (permutation-covariant) noise
%to a PI/collective-state density operator represented as blocks over J irreps.
%
% Currently implemented:
%   model = 'depolarizing'  with param.p in [0, 3/4]
%       Single-qubit: D_p(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ)
%       Implemented via Lindbladian time t with eta = 1 - 4p/3 = exp(-2 t)
%
% Input:
%   rho_c : struct from pi_code_to_collective_rho (or same fields):
%           .N, .J_vals, .blocks
%   model : string
%   param : struct, e.g. param.p = 0.1
%
% Output:
%   rho_out : same struct fields, with updated blocks and .full

    N = rho_c.N;
    J_vals = rho_c.J_vals(:).';
    blocks_in = rho_c.blocks;

    switch lower(model)
        case 'depolarizing'
            if ~isfield(param,'p'); error('param.p required for depolarizing.'); end
            p = param.p;
            if p < 0 || p > 3/4
                error('Depolarizing parameter p must satisfy 0 <= p <= 3/4.');
            end
            eta = 1 - 4*p/3;                 % Bloch shrink factor
            eta_eps = max(eta, 1e-14);       % avoid log(0)
            t = -0.5*log(eta_eps);           % choose gamma=1 so exp(-2t)=eta

            % Build superoperator for Lindbladian:
            % L(rho) = (1/2)*(2G_{+-} + 2G_{-+} + G_{zz} - 3N*rho)
            L = build_L_depolarizing(N, J_vals);

            % Vectorize input blocks, evolve, devectorize
            vin = blocks_to_vec(blocks_in, J_vals);
            vout = expm(L*t) * vin;
            blocks_out = vec_to_blocks(vout, J_vals);

            % pack
            rho_out = rho_c;
            rho_out.blocks = blocks_out;
            rho_out.full = blkdiag(blocks_out{:});

        otherwise
            error('Unsupported model: %s', model);
    end
end

% ------------------------- helper: build Lindbladian -------------------------
function L = build_L_depolarizing(N, J_vals)
    % Precompute irrep dims and offsets in operator-vector space
    numJ = numel(J_vals);
    dJ = zeros(1,numJ);
    for k=1:numJ
        dJ(k) = round(2*J_vals(k) + 1);
    end
    offsets = zeros(1,numJ);
    for k=2:numJ
        offsets(k) = offsets(k-1) + dJ(k-1)^2;
    end
    D = offsets(end) + dJ(end)^2;  % total operator-space dimension

    % Map 2J -> index
    twoJ_list = round(2*J_vals);
    Jindex = containers.Map('KeyType','int32','ValueType','int32');
    for k=1:numJ
        Jindex(int32(twoJ_list(k))) = int32(k);
    end

    % Precompute d_N^J and alpha_N^J on this J grid
    d_mult = zeros(1,numJ);
    for k=1:numJ
        d_mult(k) = dN_J(N, J_vals(k));
    end
    alpha = zeros(1,numJ);
    runsum = 0;
    for k=numJ:-1:1
        runsum = runsum + d_mult(k);
        alpha(k) = runsum;
    end

    % Build sparse superoperator
    I = [];
    J = [];
    V = [];

    % Convenience: add coefficient into sparse triplets
    function add_entry(row, col, val)
        I(end+1,1) = row;
        J(end+1,1) = col;
        V(end+1,1) = val;
    end

    % Convenience: add G_{qr} contribution for current (Jval,M,Mp)
    function add_G_term(shift_q, shift_r, col_in, pref)
        terms = g_qr_terms(N, Jval, M, Mp, shift_q, shift_r, ...
                           J_vals, dJ, offsets, Jindex, d_mult, alpha);
        for tt = 1:size(terms,1)
            add_entry(terms(tt,1), col_in, pref * terms(tt,2));
        end
    end

    % For each basis operator E_{J}(m,mp), apply L and record coefficients
    for k = 1:numJ
        Jval = J_vals(k);
        d = dJ(k);

        for mi = 1:d
            M  = -Jval + (mi-1);
            for mpi = 1:d
                Mp = -Jval + (mpi-1);

                col = offsets(k) + (mi-1)*d + mpi; % column index of this basis operator

                % Start with the -3N*rho term: (1/2)*(-3N)*E
                add_entry(col, col, 0.5 * (-3*N));

                % Add (1/2)*[2G_{+-} + 2G_{-+} + G_{zz}]
                add_G_term(+1, -1,  col, 1.0); % 0.5*2 = 1.0
                add_G_term(-1, +1,  col, 1.0); % 0.5*2 = 1.0
                add_G_term( 0,  0,  col, 0.5); % 0.5*1 = 0.5
            end
        end
    end

    L = sparse(I, J, V, D, D);
end

% ------------------------- helper: g_{qr} action terms -------------------------
function terms = g_qr_terms(N, J, M, Mp, shift_q, shift_r, ...
                           J_vals, dJ, offsets, Jindex, d_mult, alpha)
% Returns rows and coefficients for:
%   G_{qr}(|J,M><J,Mp|) = sum_n sigma_q^(n) |J,M><J,Mp| sigma_r^(n)^\dagger
% in the reduced collective operator basis (multiplicity traced out),
% using the closed-form J, J±1 coupling coefficients.

    % q,r as shifts: +1 => '+', -1 => '-', 0 => 'z'
    % Output terms: each row is [global_row_index, coeff]
    out = zeros(0,2);

    % helper to add a basis operator |Jout,Mout><Jout,Mpout|
    function push_term(Jout, Mout, Mpout, coeff)
        if abs(coeff) < 1e-15; return; end
        twoJ = int32(round(2*Jout));
        if ~isKey(Jindex, twoJ); return; end
        k = double(Jindex(twoJ));
        d = dJ(k);

        if Mout < -Jout || Mout > Jout || Mpout < -Jout || Mpout > Jout
            return;
        end

        mi  = round(Mout + Jout + 1);
        mpi = round(Mpout + Jout + 1);

        row = offsets(k) + (mi-1)*d + mpi;
        out(end+1,:) = [row, coeff];
    end

    % Identify q,r coefficient functions (A,B,D) using shifts
    Aq = Acoef(J, M,  shift_q);
    Ar = Acoef(J, Mp, shift_r);

    Bq = Bcoef(J, M,  shift_q);
    Br = Bcoef(J, Mp, shift_r);

    Dq = Dcoef(J, M,  shift_q);
    Dr = Dcoef(J, Mp, shift_r);

    % Precompute multiplicity/alpha factors at this J
    twoJ = int32(round(2*J));
    kJ = double(Jindex(twoJ));
    dJmult = d_mult(kJ);

    % alpha(J) is alpha at this J grid, alpha(J+1) needs lookup if exists
    alphaJ = alpha(kJ);
    alphaJp1 = 0;
    if isKey(Jindex, int32(round(2*(J+1))))
        alphaJp1 = alpha(double(Jindex(int32(round(2*(J+1))))));
    end

    % Shifts in M and Mp
    Mout_same  = M  + shift_q;
    Mpout_same = Mp + shift_r;

    % ---- Same-J term coefficient C^(J) ----
    % For J==0, set same-J and (J-1) terms to 0 (numerators also vanish).
    if J > 0
        Csame = (Aq * Ar) / (2*J) * (1 + alphaJp1 * (2*J+1) / (dJmult * (J+1)));
        push_term(J, Mout_same, Mpout_same, Csame);
    end

    % ---- J-1 term coefficient C^(J-1) ----
    if J > 0 && isKey(Jindex, int32(round(2*(J-1))))
        Cminus = (alphaJ) / (2*J * dJmult) * (Bq * Br);
        push_term(J-1, Mout_same, Mpout_same, Cminus);
    end

    % ---- J+1 term coefficient C^(J+1) ----
    if isKey(Jindex, int32(round(2*(J+1))))
        Cplus  = (alphaJp1) / (2*(J+1) * dJmult) * (Dq * Dr);
        push_term(J+1, Mout_same, Mpout_same, Cplus);
    end

    terms = out;
end

% ------------------------- A,B,D coefficients -------------------------
function v = Acoef(J, M, sh)
    if sh == +1
        v = sqrt((J - M) * (J + M + 1));
    elseif sh == -1
        v = sqrt((J + M) * (J - M + 1));
    else % sh == 0 (z)
        v = M;
    end
end

function v = Bcoef(J, M, sh)
    if sh == +1
        v = sqrt((J - M) * (J - M - 1));
    elseif sh == -1
        v = -sqrt((J + M) * (J + M - 1));
    else % sh == 0
        v = sqrt((J + M) * (J - M));
    end
end

function v = Dcoef(J, M, sh)
    if sh == +1
        v = -sqrt((J + M + 1) * (J + M + 2));
    elseif sh == -1
        v = sqrt((J - M + 1) * (J - M + 2));
    else % sh == 0
        v = sqrt((J + M + 1) * (J - M + 1));
    end
end

% ------------------------- multiplicity d_N^J -------------------------
function d = dN_J(N, J)
% d_N^J = N! (2J+1) / ((N/2 - J)! (N/2 + J + 1)!)
% Use gamma for half-integers.
    d = exp(gammaln(N+1) + log(2*J+1) ...
        - gammaln(N/2 - J + 1) ...
        - gammaln(N/2 + J + 2));
end

% ------------------------- vectorization helpers -------------------------
function v = blocks_to_vec(blocks, J_vals)
    numJ = numel(J_vals);
    v = [];
    for k=1:numJ
        B = blocks{k};
        v = [v; B(:)];
    end
end

function blocks = vec_to_blocks(v, J_vals)
    numJ = numel(J_vals);
    blocks = cell(1,numJ);
    pos = 1;
    for k=1:numJ
        d = round(2*J_vals(k) + 1);
        n = d*d;
        blocks{k} = reshape(v(pos:pos+n-1), [d,d]);
        pos = pos + n;
    end
end
