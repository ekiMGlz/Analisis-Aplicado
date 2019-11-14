function [xf, iter] = GaussNewton(r, x0, tol, maxiter, J)
%GaussNewton - Description
%
% Syntax: output = GaussNewton(f, x0, tol, maxite)
%
% Long description
    f = @(x) 0.5*dot(r(x), r(x));
    iter = 0;
    xf = x0;

    gk = inf;

    while norm(gk, 'inf') > tol && iter < maxiter
        Jk = J(xf);
        rk = r(xf);
        gk = Jk'*rk;

        dk = -(Jk'*Jk)\gk;

        [alpha, ~] = encAlpha(f, xf, dk, gk);

        if alpha*norm(dk, 'inf') < 1e-7*norm(xf, 'inf')
            break;
        end

        xf = xf + alpha*dk;
        iter  = iter + 1;
    end
end