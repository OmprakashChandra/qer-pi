function reshaped_eigenvectors = recovery_new(X, logical0, logical1)
    cutoff = 1e-10;

    if issparse(X), X = full(X); end

    d   = length(logical0);
    d_l = 2;

    % Encoder: columns are |0_L>, |1_L>
    Encoder = [logical0, logical1]; % (d x d_l)

    % Make X explicitly Hermitian (removes tiny solver asymmetry)
    X = (X + X')/2;

    % Eigendecomposition for Hermitian X
    [V, lam] = eig(X, "vector");   % lam is vector of eigenvalues
    lam = real(lam);

    % Keep only significant nonnegative eigenvalues
    keep = find(lam > cutoff);
    reshaped_eigenvectors = cell(1, numel(keep));

    for ii = 1:numel(keep)
        idx = keep(ii);

        % Reshape vec -> (d_l x d)
        R = reshape(V(:, idx), [d_l, d]);

        % Kraus in Dicke subspace (d x d): Encoder (d x d_l) * R (d_l x d)
        reshaped_eigenvectors{ii} = sqrt(lam(idx)) * (Encoder * R);
    end
end