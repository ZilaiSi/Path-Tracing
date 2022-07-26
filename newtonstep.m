function [step,preconditioner] = newtonstep(x,phi,r,eta,A,b,preconditioner,theta_0)

%%% Outputs
% 1. step: Newton step
% 2. preconditioner.it: number of iterations of krylov solver
% 3. preconditioner.flag: whether the krylov solver converge or not.

%%% Inputs
% 1. x, the solution for a particular set of hyperparameters
% 2. phi: log varinace (log(theta))
% 3. r: the current value of hyperparemeter r
% 4. eta: the current value of hyperparemeter r
% 5. A: the current value of hyperparemeter theta_0
% 6. b: the blurred signal
% 7. precondtioner: preconditioner for krylov solver
% 8. theta_0: the current value of hyperparemeter theta_0

%% get dimensions
[~,n] = size(A);

%% Build fidelity and penalty terms
theta = exp(phi);
D_theta = sparse((1:n),(1:n),theta,n,n);
a = (D_theta.^(1/2))*(A'*A)*(D_theta.^(1/2));
rhs = - (theta.^(-1/2)).*x;
c = -(1/2)*(theta.^(-1)).*x.^2 + theta_0^(-r)*(r^2)*theta.^r;
%
H_A = sparse([],[],[],2*n,2*n);
H_A((1:n),(1:n)) = a;

R = sparse((1:2*n),(1:2*n),1,2*n,2*n) + sparse((1:n),n + (1:n),rhs,2*n,2*n);
S = sparse((1:2*n),(1:2*n),[ones([n,1]);c],2*n,2*n);

H_P = R'*S*R;

%% Build Hessian
H = H_A + H_P;

%% Build right hand side (negative gradient of cost functio0n)
discrepancy = A*x - b;
g = [A'*discrepancy + theta.^(-1).*x; -(1/2)*theta.^(-1).*(x.^2) - eta + theta_0^(-r)*r*theta.^r];
rhs = -g;

rhs = diag([theta.^(1/2);ones(1,n)'])*rhs;
tol = 0.5;
[P,~,~] = approximate_Preconditioner(A,x,theta,r,0,tol,theta_0);

%% store for next time we need to update
preconditioner.H_old = H; % old Hessian
preconditioner.P_old = P; % old preconditioner

H_pre = @(z) P(H*z);
rhs_pre = P(rhs);

max_it = min(2*n,40);

[y,exit_flag,~,iterations] = cgs(H_pre,rhs_pre,10^(-6),max_it);

step = nan([2*n,1]);
step(1:n) = (theta.^(1/2)).*y((1:n));
step(n+1:2*n) = y((n+1:2*n));

%% store preconditioner performance
preconditioner.it = iterations;
preconditioner.flag = exit_flag;
end