A = pascal(4);
b = ones(4, 1);
f = @(x) 0.5*x'*A*x + dot(b, x) + 1;
x0 = [4; 4; 4; 4];

[xf, iters] = lineBGFS(f, x0, 1e-5, 1000)