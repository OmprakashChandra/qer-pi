function rho_c = col_rho(log0, log1)
%PI_CODE_TO_COLLECTIVE_RHO  Build the collective (irrep-block) density matrix
%from PI codewords supported on the Dicke (symmetric) subspace.
%
% Inputs:
%   log0, log1 : (N+1)x1 complex column vectors in Dicke basis
%                ordered as w = 0,1,...,N  (i.e., M = w - N/2)
%
% Output (struct):
%   rho_c.N        : number of qubits N
%   rho_c.J_vals   : list of total-spin irreps [N/2, N/2-1, ..., Jmin]
%   rho_c.blocks   : cell array of density blocks for each J in rho_c.J_vals
%                    (only J=N/2 block is nonzero for PI code input)
%   rho_c.full     : full block-diagonal collective density matrix (sum_J (2J+1))^2
%
% By default we form the maximally mixed state on the codespace:
%   rho = (|0_L><0_L| + |1_L><1_L|)/2
%
% Notes:
%   - This assumes the codewords are fully supported in the symmetric subspace (J=N/2),
%     i.e., Dicke basis. Hence only the J=N/2 block is populated initially.

    % --- basic checks / shaping ---
    log0 = log0(:);
    log1 = log1(:);

    if length(log0) ~= length(log1)
        error('log0 and log1 must have the same length (N+1).');
    end

    N = length(log0) - 1;
    if N < 1 || abs(N - round(N)) > 0
        error('Input length must be N+1 with integer N >= 1.');
    end

    % --- form maximally mixed state on codespace in Dicke basis ---
    rhoJmax = 0.5*(log0*log0' + log1*log1');
    rhoJmax = (rhoJmax + rhoJmax')/2;  % enforce Hermitian numerically

    % --- build list of J irreps for N spins-1/2 ---
    Jmax = N/2;
    if mod(N,2)==0
        Jmin = 0;
    else
        Jmin = 0.5;
    end
    J_vals = Jmax:-1:Jmin;  % step -1 (works for half-integers too)

    % --- allocate blocks: only J=N/2 is nonzero for PI code input ---
    numJ = numel(J_vals);
    blocks = cell(1, numJ);

    for k = 1:numJ
        J = J_vals(k);
        dJ = round(2*J + 1);  % dimension of spin-J irrep
        if k == 1
            % J = Jmax block matches Dicke basis dimension (N+1)
            if dJ ~= (N+1)
                error('Internal mismatch: dim(Jmax) ~= N+1.');
            end
            blocks{k} = rhoJmax;
        else
            blocks{k} = zeros(dJ, dJ);
        end
    end

    % --- full block-diagonal collective density matrix (optional but handy) ---
    fullMat = blocks{1};
    for k = 2:numJ
        fullMat = blkdiag(fullMat, blocks{k});
    end

    % --- pack output ---
    rho_c = struct();
    rho_c.N      = N;
    rho_c.J_vals = J_vals;
    rho_c.blocks = blocks;
    rho_c.full   = fullMat;
end
