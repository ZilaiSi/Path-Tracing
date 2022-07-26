function [dxdt,dphidt,y,preconditioner] = Compute_Derivatives(x,phi,r,theta_0,drdt,detadt,dtheta_0dt,A,preconditioner,y_old)
%% compute dx/dt and dtheta/dt given r, eta, dr/dt, deta/dt, and A

%%% Outputs
% 1. dxdt: computed change rate of x
% 2. dphidt: computed change rate of phi
% 3. y: solution of ODE
% 4. preconditioner.it: number of iterations of krylov solver
% 5. preconditioner.flag: whether the krylov solver converge or not.

%%% Inputs
% 1. x, the solution for a particular set of hyperparameters
% 2. phi: log varinace (log(theta))
% 3. r, the current value of hyperparemeter r
% 4. theta_0: the current value of hyperparemeter theta_0
% 4, drdt the current value of hyperparemeter eta
% 5. detadt: the current value of hyperparemeter theta_0
% 5. dtheta_0dt: the current value of hyperparemeter theta_0
% 5. A: the current value of hyperparemeter theta_0
% 5. m, the order to carry the preconditioner expansion out to, likely
% only want order = 1, much beyond that may become unstable
% 6. tol, the tolerance for the low rank approximation to the fidelity term
% 7. A: forward model A
% 8. precondtioner: preconditioner for krylov solver
% 9. y_old: solution of last ODE system used as initial value

%% get dimensions
[~,n] = size(A);

%% Build fidelity and penalty terms
theta = exp(phi);
D_theta = sparse((1:n),(1:n),theta,n,n);
a = (D_theta.^(1/2))*(A'*A)*(D_theta.^(1/2));
rhs = - (theta.^(-1/2)).*x;
c = -(1/2)*(theta.^(-1)).*x.^2 + theta_0^(-r)*(r^2)*theta.^r;
H_A = sparse([],[],[],2*n,2*n);
H_A((1:n),(1:n)) = a;
R = sparse((1:2*n),(1:2*n),1,2*n,2*n) + sparse((1:n),n + (1:n),rhs,2*n,2*n);
S = sparse((1:2*n),(1:2*n),[ones([n,1]);c],2*n,2*n);
H_P = R'*S*R;

%% Build Hessian
H = H_A + H_P;

%% Compute right hand side
rhs = -sparse((n+1:2*n),1,1/(theta_0^r)*(1 + r*log(theta/theta_0)).*(theta.^r),2*n,1)*drdt + ...
    sparse((n+1:2*n),1,1,2*n,1)*detadt + ...
    sparse((n+1:2*n),1,r^2/theta_0^(r+1)*theta.^(r),2*n,1)*dtheta_0dt;

rhs = diag([theta.^(1/2);ones(1,n)'])*rhs;
%% recompute using Sherman-Morrison and 1st order perturbation correction
tol = 1e-3; % 1d
[P,~,~,] = approximate_Preconditioner(A,x,theta,r,0,tol,theta_0); % m = 0
%% store for next time we need to update
preconditioner.H_old = H; % old Hessian
preconditioner.P_old = P; % old preconditioner

H_pre = @(z) P(H*z);
rhs_pre = P(rhs);
%% Solve the linear system iteratively

max_it = min(2*n,60);

if isnan(y_old)
    [y,exit_flag,~,iterations] = cgs(H_pre,rhs_pre,10^(-6),max_it);
else
    %tic
    [y,exit_flag,~,iterations] = cgs(H_pre,rhs_pre,10^(-6),max_it,[],[],y_old);
end

%% rescale and change variables
dxdt = (theta.^(1/2)).*y((1:n));
dphidt = y((n+1:2*n));

%% store preconditioner performance
preconditioner.it = iterations;
preconditioner.flag = exit_flag;
end