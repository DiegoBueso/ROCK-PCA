%% Promax rotation
% this is a simplification of MathWorks "rotatefactors" function to
% optimize the method.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright 1993-2014 The MathWorks, Inc.
%   References:
%     [1] Harman, H.H. (1976) Modern Factor Analysis, 3rd Ed., University
%         of Chicago Press.
%     [2] Lawley, D.N. and Maxwell, A.E. (1971) Factor Analysis as a
%         Statistical Method, 2nd Ed., American Elsevier Pub. Co.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function B = Promax(A,power)

[d, m] = size(A);

B0 = orthomax(A);
B = sign(B0) .* abs(B0).^power;

function [B, T] = orthomax(A)
[d, m] = size(A);
gamma=1;
reltol = sqrt(eps(class(A)));
maxit = 250;

h = sqrt(sum(A.^2, 2));
A = bsxfun(@rdivide, A, h);

% De facto, the intial rotation matrix is identity.
T = eye(m);
B = A * T;

converged = false;
if (0 <= gamma) && (gamma <= 1)
    % Use Lawley and Maxwell's fast version

    % Choose a random rotation matrix if identity rotation 
    % makes an obviously bad start.
    [L, ~, M] = svd(A' * (d*B.^3 - gamma*B * diag(sum(B.^2))));
    T = L * M';
    if norm(T-eye(m)) < reltol
        % Using identity as the initial rotation matrix, the first 
        % iteration does not move the loadings enough to escape the 
        % the convergence criteria.  Therefore, pick an initial rotation
        % matrix at random.
        [T,~] = qr(randn(m,m));
        B = A * T;
    end
    
    D = 0;
    for k = 1:maxit
        Dold = D;
        [L, D, M] = svd(A' * (d*B.^3 - gamma*B * diag(sum(B.^2))));
        T = L * M';
        D = sum(diag(D));
        B = A * T;
       if abs(D - Dold)/D < reltol
            converged = true;
            break;
        end
    end
end

if ~converged
    %error(message('stats:rotatefactors:IterationLimit'));
    B=NaN(size(B));
end

% Unnormalize the rotated loadings
B = bsxfun(@times, B, h);
end

end