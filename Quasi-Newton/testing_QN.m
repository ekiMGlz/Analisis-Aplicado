%A = pascal(4);
%b = -ones(4, 1);
%f = @(x) 0.5*x'*A*x + dot(b, x) + 1;
f = @(x) 100*(x(1)^2 - x(2))^2 + (x(1)-1)^2;
x0 = [2; 3];

%[xf, iters] = lineBGFS(f, x0, 1e-5, 1000)
%[xf, msg] = TRSR1(f, x0, 1000, 1e-5)
[xf, iters] = limBGFS(f, x0, 1e-5, 1000, 3)
[xf, iters] = limBGFS_cyclic(f, x0, 1e-5, 1000, 3)