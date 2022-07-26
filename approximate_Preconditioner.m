function [P,effective_rank,output_flag] = approximate_Preconditioner(A,x,theta,r,m,tol,theta_0)
%% Approximates the preconditioner for the rescaled, post change of variables Hessian

% Outputs:
% 1. P, the approximate preconditioner
% 2. effective_rank, the rank of the approximation to the fidelity term
% 3. output_flag, message about whether H_A - U*Sigma U' has spectral radius < 1 or > 1

% Inputs:
% 1. A, the forward model
% 2. x, the solution for a particular set of hyperparameters
% 3. theta, the solution variances for a particular set of hyperparameters
% 4. r, the current value of hyperparemeter r
% 5. m, the order to carry the preconditioner expansion out to, likely
% only want order = 1, much beyond that may become unstable
% 6. tol, the tolerance for the low rank approximation to the fidelity term
% 7. the current value of hyperparemeter theta_0s

%% get dimensions
[~,n] = size(A);

%% Build fidelity and penalty terms
%theta = exp(phi);
D_theta = sparse((1:n),(1:n),theta,n,n);
a = (D_theta.^(1/2))*(A'*A)*(D_theta.^(1/2));
rhs = - (theta.^(-1/2)).*x;
c = -(1/2)*(theta.^(-1)).*x.^2 + theta_0^(-r)*(r^2)*theta.^r;
%% Build inverse of Penalty Term
R_inv = sparse((1:2*n),(1:2*n),1,2*n,2*n) + sparse((1:n),n + (1:n),-rhs,2*n,2*n);
S_inv = sparse((1:2*n),(1:2*n),[ones([n,1]);c.^(-1)],2*n,2*n);

H_P_inv = R_inv*S_inv*R_inv';

%% Get low rank approximation to fidelity term

% svd based approach, iteratively adds terms from svd until ||U Sigma U' -
% a||_{fro} < tol*||a||_{fro}
% gershgorin based approach
column_sums = sum(abs(a));
%columns_to_keep = find(column_sums >= 1);
columns_to_keep = find(column_sums >= tol/2);
a_trunc = a(columns_to_keep,columns_to_keep);
[U_trunc,Sigma,~] = svdsketch(a_trunc,tol/2);
U_trunc = U_trunc*Sigma.^(1/2);
[~,effective_rank] = size(U_trunc);

U = sparse([],[],[],n,effective_rank);
for j = 1:effective_rank
    U = U + sparse(columns_to_keep,j,U_trunc(:,j),n,effective_rank);
end

% add bottom block of zeros
U = [U;sparse([],[],[],n,effective_rank)];

%% Woodbury to approximate preconditioner
P = H_P_inv;
W = inv(eye([effective_rank,effective_rank]) + U'*P*U); % is size of effective rank by effective rank, may be slow or unstable if the effective rank is too large
PU = P*U;
% v = PU*W;

P = @(z) P*z - PU*(W*(PU'*z));

%% Improve preconditioner via perturbation expansion
if m > 0
    E = sparse([],[],[],2*n,2*n);
    E((1:n),(1:n)) = a  - U((1:n),:)*U((1:n),:)'; % error in low rank approximation of fidelity term
    D = -P*E;
    
    % check that the largest eigenvalue of E is less than 1 in magnitude
    lambda = eigs(D,1);s
    if abs(lambda) < 1
        output_flag = 'Spectral Radius of Discrepancy < 1, P is Sherman-Morrison corrected by perturbation expansion';
        p = P;
        for j = 1:m
            P = p + D*P;
        end
        
    else
        output_flag = 'Spectral Radius of Discrepancy >= 1, P is Sherman-Morrison';
    end
    
else
    output_flag = 'P is Sherman-Morrison';
    
end
end