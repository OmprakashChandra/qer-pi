function out = state_prepe_gpg(M, Pulses, varargin)
    
    % Parameters
    % ____________________
    % M      : integer, number of qubits
    % Pulses : integer, number of pulses in the sequence
    % ____________________
    % Returns
    % out : struct with fields:
    %   xbest      : best parameters (4*Pulses x 1)
    %   Fbest      : fidelity |<psiT|psi>|^2
    %   fbest      : infidelity-like: 1 - Fbest
    %   objbest    : least-squares objective: 2(1 - |<..>|)
    %   psi_best   : achieved state
    %   psi_target : target state
    %   rge        : m values
    %   X          : Pulses x 4 matrix [alpha beta gamma kappa]
    %   info       : settings and diagnostics
    % ____________________
%STATE_PREPE_GPG  Robust state-prep in the Dicke subspace.
%
% This is a self-contained, "best-practice" optimizer for your control ansatz:
%   Per pulse n:  U_n = Rz(alpha_n) * Ry(beta_n) * Rz(gamma_n) * G(kappa_n)
%   where G(kappa) = diag(exp(-i*kappa*m^2)) in Dicke basis m=-S..S, S=M/2.
%
% Quick overview (how we reach the target from a given initial state):
%   1) Start from |psi0> in the Dicke basis (user-provided or default |m=-S>).
%   2) Apply a sequence of P pulses; each pulse combines nonlinear phase
%      shaping G(kappa) with collective rotations Rz-Ry-Rz.
%   3) After each candidate sequence, compare the generated state to
%      |psi_target> after removing only the global phase.
%   4) Optimize pulse parameters [alpha, beta, gamma, kappa] to minimize
%      this phase-aligned distance, which is equivalent to maximizing fidelity.
%   5) Use continuation (P=1..P) plus multistart/random restarts to avoid
%      poor local minima and improve the final overlap.
%
% Target state:
%   User-supplied state vector in the Dicke basis of length M+1.
%   If omitted, defaults to the uniform Dicke superposition.
% Initial state:
%   User-supplied state vector in the Dicke basis of length M+1.
%   If omitted, defaults to |m=-S>.
%
% Optimization strategy ("best" in practice):
%   1) Continuation in pulse count: solve P=1..Pulses progressively.
%   2) Bounded local optimization with fmincon (SQP) if available.
%   3) MultiStart (if available) or random-restart wrapper.
%   4) Wrapped parameters inside objective (periodicity handled cleanly).
%   5) Smooth least-squares objective with optimal global phase removed:
%        f = || psi(x) - e^{i phi} psi_target ||^2 = 2(1 - |<psiT|psi>|)
%
% Usage:
%   out = state_prepe_gpg(M, P);
%   out = state_prepe_gpg(M, P, target_state);
%   out = state_prepe_gpg(M, P, target_state, 'InitialState', psi0);
%   out = state_prepe_gpg(M, P, target_state, 'Restarts', 200, 'UseMultiStart', true);
%
% Name-Value options:
%   'TargetState'   (default [])    Dicke-basis target state coefficients
%   'InitialState'  (default [])    Dicke-basis initial state coefficients
%   'Restarts'      (default 200)   number of start points per continuation stage
%   'MaxIters'      (default 1500)  max fmincon iterations per local solve
%   'Display'       (default 'off') 'off'|'iter'
%   'Seed'          (default [])    RNG seed
%   'UseMultiStart' (default true)  if MultiStart exists, use it
%   'KappaMax'      (default pi)    bound for kappa in [-KappaMax, KappaMax]
%   'BetaMax'       (default pi)    bound for beta  in [-BetaMax, BetaMax]
%   'AngleMax'      (default pi)    bound for alpha,gamma in [-AngleMax, AngleMax]
%   'Continuation'  (default true)  whether to do progressive P=1..Pulses
%
% Output:
%   out.xbest     - best parameters (4*Pulses x 1)
%   out.Fbest     - fidelity |<psiT|psi>|^2
%   out.fbest     - infidelity-like: 1 - Fbest
%   out.objbest   - least-squares objective: 2(1 - sqrt(Fbest))? (actually 2(1-|<..>|))
%   out.psi_best  - achieved state
%   out.psi_target
%   out.rge       - m values
%   out.X         - Pulses x 4 matrix [alpha beta gamma kappa]
%   out.info      - settings and diagnostics
%
% Notes:
%   - Requires Optimization Toolbox for fmincon; MultiStart is optional.
%   - If fmincon is unavailable, it falls back to fminsearch (still with wrapping + continuation).
%
% Gavin-style warning: For larger M, you may need more restarts and/or pulses.

target_state = [];
initial_state = [];
if ~isempty(varargin)
    firstArg = varargin{1};
    if isnumeric(firstArg) || islogical(firstArg) || isa(firstArg, 'sym')
        target_state = firstArg;
        varargin = varargin(2:end);
    end
end

% -------------------- Parse options --------------------
p = inputParser;
p.addParameter('TargetState', target_state, @(z) isempty(z) || isnumeric(z) || islogical(z));
p.addParameter('InitialState', initial_state, @(z) isempty(z) || isnumeric(z) || islogical(z));
p.addParameter('Restarts', 200, @(z) isnumeric(z) && isscalar(z) && z>=1);
p.addParameter('MaxIters', 1500, @(z) isnumeric(z) && isscalar(z) && z>=1);
p.addParameter('Display', 'off', @(s) ischar(s) || isstring(s));
p.addParameter('Seed', [], @(z) isempty(z) || (isnumeric(z) && isscalar(z)));
p.addParameter('UseMultiStart', true, @(b) islogical(b) && isscalar(b));
p.addParameter('KappaMax', pi, @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('BetaMax', pi, @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('AngleMax', pi, @(z) isnumeric(z) && isscalar(z) && z>0);
p.addParameter('Continuation', true, @(b) islogical(b) && isscalar(b));
p.parse(varargin{:});
opt = p.Results;

if ~isempty(opt.Seed)
    rng(opt.Seed);
end

% -------------------- Dimensions / basis --------------------
S   = M/2;
rge = -S:S;               % m values
d   = M + 1;

psi0_default = zeros(d,1);
psi0_default(1) = 1;      % |m=-S>

psi_target_default = ones(d,1) / sqrt(d);

psi0 = opt.InitialState;
if isempty(psi0)
    psi0 = psi0_default;
else
    psi0 = psi0(:);
    if numel(psi0) ~= d
        error('InitialState must have length %d for M=%d.', d, M);
    end
    nrm0 = norm(psi0);
    if nrm0 == 0
        error('InitialState must be nonzero.');
    end
    psi0 = psi0 / nrm0;
end

psi_target = opt.TargetState;
if isempty(psi_target)
    psi_target = psi_target_default;
else
    psi_target = psi_target(:);
    if numel(psi_target) ~= d
        error('TargetState must have length %d for M=%d.', d, M);
    end
    nrmT = norm(psi_target);
    if nrmT == 0
        error('TargetState must be nonzero.');
    end
    psi_target = psi_target / nrmT;
end

% -------------------- Build Rx(pi/2) helper (same convention as your code) --------------------
A = zeros(d,d);
for j = -S:(S-1)
    idx = j + S + 1;
    c = sqrt(S*(S+1) - j*(j+1));
    A(idx, idx+1) = c;
    A(idx+1, idx) = c;
end
Rx_pi2 = expm(1i*(pi/4)*A);  % = R_x(pi/2) (with your 2Jx convention)

% Gate constructors
Rz = @(theta) diag(exp(-1i*theta*rge));
Ry = @(theta) Rx_pi2 * Rz(theta) * Rx_pi2';
G  = @(kappa) diag(exp(-1i*kappa*(rge.^2)));

% Wrapping helper (keeps optimizer in one fundamental domain)
wrap = @(t) mod(t+pi, 2*pi) - pi;

% -------------------- Apply pulse sequence --------------------
    function psi_out = apply_sequence(x, P)
        psi = psi0;
        for n = 1:P
            a = x(4*n-3); b = x(4*n-2); g = x(4*n-1); k = x(4*n);
            % wrap periodic params (important!)
            a = wrap(a);  b = wrap(b);  g = wrap(g);  k = wrap(k);
            % same order as your density-matrix update: phase then rotation
            psi = G(k) * psi;
            psi = (Rz(a) * Ry(b) * Rz(g)) * psi;
        end
        psi_out = psi / norm(psi);
    end

% Smooth least-squares objective with best global phase removed
    function f = obj_ls(x, P)
        psi = apply_sequence(x, P);
        c = psi_target' * psi;
        phi = angle(c);
        diff = psi - exp(1i*phi) * psi_target;
        f = real(diff' * diff);   % = 2(1 - |<psiT|psi>|)
    end

% -------------------- Choose optimizer availability --------------------
has_fmincon = exist('fmincon','file') == 2;
has_ms      = exist('MultiStart','class') == 8 || exist('MultiStart','file') == 2; % robust-ish
use_ms      = opt.UseMultiStart && has_fmincon && has_ms;

% Bounds per pulse: alpha,gamma in [-AngleMax,AngleMax], beta in [-BetaMax,BetaMax], kappa in [-KappaMax,KappaMax]
    function [lb, ub] = bounds_for(P)
        lb = zeros(4*P,1); ub = zeros(4*P,1);
        lb(1:4:end) = -opt.AngleMax;  ub(1:4:end) =  opt.AngleMax;  % alpha
        lb(2:4:end) = -opt.BetaMax;   ub(2:4:end) =  opt.BetaMax;   % beta
        lb(3:4:end) = -opt.AngleMax;  ub(3:4:end) =  opt.AngleMax;  % gamma
        lb(4:4:end) = -opt.KappaMax;  ub(4:4:end) =  opt.KappaMax;  % kappa
    end

% Local solve wrapper for a given P and start x0
    function [xsol, fsol, exitflag, output] = local_solve(P, x0)
        if has_fmincon
            [lb, ub] = bounds_for(P);
            f = @(x) obj_ls(x, P);
            options_fmincon = optimoptions('fmincon', ...
                'Algorithm', 'sqp', ...
                'Display', char(opt.Display), ...
                'MaxIterations', opt.MaxIters, ...
                'MaxFunctionEvaluations', max(5000, 50*(4*P)), ...
                'OptimalityTolerance', 1e-12, ...
                'StepTolerance', 1e-12);
            problem = createOptimProblem('fmincon', ...
                'objective', f, ...
                'x0', x0, ...
                'lb', lb, ...
                'ub', ub, ...
                'options', options_fmincon);

            if use_ms
                ms = MultiStart('Display','off', ...
                                'UseParallel', false, ...
                                'StartPointsToRun','bounds');
                % For MultiStart we treat x0 as one of many starts; caller will handle start sets.
                [xsol, fsol, exitflag, output] = fmincon(problem);
            else
                [xsol, fsol, exitflag, output] = fmincon(problem);
            end
        else
            % Fallback: fminsearch (unconstrained) with wrapping in objective
            f = @(x) obj_ls(x, P);
            options_fs = optimset('Display', char(opt.Display), 'MaxIter', opt.MaxIters);
            [xsol, fsol, exitflag, output] = fminsearch(f, x0, options_fs);
        end
    end

% -------------------- Continuation loop --------------------
Pfinal = Pulses;
xbest_full = zeros(4*Pfinal,1);
best_trace = struct('P',{},'obj',{},'F',{},'exitflag',{});

Pstart = 1;
if ~opt.Continuation
    Pstart = Pfinal;
end

for P = Pstart:Pfinal
    dim = 4*P;

    % Construct initial guess:
    %   - If continuation, pad previous solution with zeros for new pulse.
    if P == Pstart
        x_seed = zeros(dim,1);
    else
        x_seed = zeros(dim,1);
        x_seed(1:4*(P-1)) = xbest_full(1:4*(P-1));
        % new pulse initialization: mild random
        x_seed(4*P-3:4*P) = [0;0;0;0];
    end

    % Generate restart start points around the seed plus random points in bounds
    starts = cell(opt.Restarts,1);
    starts{1} = x_seed;

    if has_fmincon
        [lb, ub] = bounds_for(P);
        for r = 2:opt.Restarts
            xr = lb + (ub-lb).*rand(dim,1);
            % bias a fraction near the current seed
            if rand < 0.5
                xr = x_seed + 0.25*(ub-lb).*randn(dim,1);
                xr = max(lb, min(ub, xr));
            end
            starts{r} = xr;
        end
    else
        % Without bounds, still keep typical scale reasonable
        for r = 2:opt.Restarts
            xr = zeros(dim,1);
            xr(1:4:end) = pi*randn(P,1);
            xr(2:4:end) = pi*randn(P,1);
            xr(3:4:end) = pi*randn(P,1);
            xr(4:4:end) = 0.5*randn(P,1);
            if rand < 0.5
                xr = x_seed + 0.25*randn(dim,1);
            end
            starts{r} = xr;
        end
    end

    % Run either MultiStart or manual restarts
    xbest_P = [];
    fbest_P = inf;
    exitflag_P = NaN;
    output_P = [];

    if use_ms
        % Build a problem once, then pass custom start points to MultiStart
        f = @(x) obj_ls(x, P);
        [lb, ub] = bounds_for(P);
        options_fmincon = optimoptions('fmincon', ...
            'Algorithm', 'sqp', ...
            'Display', 'off', ...
            'MaxIterations', opt.MaxIters, ...
            'MaxFunctionEvaluations', max(5000, 50*(dim)), ...
            'OptimalityTolerance', 1e-12, ...
            'StepTolerance', 1e-12);
        problem = createOptimProblem('fmincon', ...
            'objective', f, 'x0', starts{1}, 'lb', lb, 'ub', ub, 'options', options_fmincon);

        % CustomStartPointSet expects rows = start points, columns = variables.
        startMat = zeros(opt.Restarts, dim);
        for rr = 1:opt.Restarts
            startMat(rr, :) = starts{rr}(:).';
        end
        sp = CustomStartPointSet(startMat);
        ms = MultiStart('Display','off', 'UseParallel', false);
        [xbest_P, fbest_P, exitflag_P, ~, ~] = run(ms, problem, sp);
    else
        for r = 1:opt.Restarts
            x0 = starts{r};
            [xloc, floc, eflag, outloc] = local_solve(P, x0);
            if floc < fbest_P
                fbest_P = floc;
                xbest_P = xloc;
                exitflag_P = eflag;
                output_P = outloc;
            end
        end
    end

    % Update the full-length parameter vector
    xbest_full(1:dim) = xbest_P(:);
    if P < Pfinal
        xbest_full(dim+1:end) = 0;
    end

    % Diagnostics for this stage
    psiP = apply_sequence(xbest_P, P);
    FP = abs(psi_target' * psiP)^2;
    best_trace(end+1) = struct('P',P,'obj',fbest_P,'F',FP,'exitflag',exitflag_P); %#ok<AGROW>

    if strcmpi(opt.Display,'iter')
        fprintf('P=%d: obj=%.3e, F=%.12f, 1-F=%.3e\n', P, fbest_P, FP, 1-FP);
    end
end

% Final evaluation
psi_best = apply_sequence(xbest_full(1:4*Pfinal), Pfinal);
Fbest = abs(psi_target' * psi_best)^2;
fbest = 1 - Fbest;
objbest = obj_ls(xbest_full(1:4*Pfinal), Pfinal);

% Pack outputs
out = struct();
out.xbest = xbest_full(1:4*Pfinal);
out.X     = reshape(out.xbest, 4, Pfinal).'; % Pulses x 4 [alpha beta gamma kappa]
out.psi_best = psi_best;
out.psi_target = psi_target;
out.rge = rge;
out.Fbest = Fbest;
out.fbest = fbest;
out.objbest = objbest;

out.info = struct();
out.info.M = M;
out.info.Pulses = Pulses;
out.info.Restarts = opt.Restarts;
out.info.MaxIters = opt.MaxIters;
out.info.has_fmincon = has_fmincon;
out.info.used_MultiStart = use_ms;
out.info.bounds = struct('AngleMax',opt.AngleMax,'BetaMax',opt.BetaMax,'KappaMax',opt.KappaMax);
out.info.continuation = opt.Continuation;
out.info.trace = best_trace;

end