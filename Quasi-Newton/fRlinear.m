function [r, J, nump, model] = fRlinear(ts, ys);
	
	nump = 3;
	r = @(x)  x(1) + x(2)*ts + x(3)*ts.^2 - ys;
	J = @(x)  [ones(size(ts)), ts, ts.^2];

	model = @(x, t)  x(1) + x(2)*t + x(3)*t.^2;
end
