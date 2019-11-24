[ts, ys] = getpts()
[r, J, nump, model] = fRlinear(ts, ys);
x0 = ones(nump,1);
tol = 1e-5;
maxIter = 500;
[xk, iter] = GaussNewton( r, x0, tol, maxIter, J )

Df = J(xk)'*r(xk);
gradfApprox = norm(Df, 'inf')


%% plot
scatter( ts, ys )
hold on
errL2  = norm(r(xk))
errinf = norm(r(xk), 'inf')
fplot( @(t)  model(xk, t), [min(ts), max(ts)] )
hold off