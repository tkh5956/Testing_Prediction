rm(list = ls())

############################################################
##    Simulation code for finite performance evaluation   ##
############################################################
##                                                        ##    
## Goal: Clustered binary classification simulation for   ##
##       superiority / noninferiority power and sample    ## 
##       size validation                                  ##
## Metric: F1 score (sufficient for demonstration)        ##  
############################################################


## ---------------------------------------------------------
## 0. Utility function
## Note: Notation for p = (p11, p10, p01, p00) in this code 
## ---------------------------------------------------------

## Logistic inverse link used to generate cluster-specific event rates
expit <- function(x) 1 / (1 + exp(-x))

confusion_counts <- function(y, yhat) {
  c(
    sum(y == 0 & yhat == 0),  # TN
    sum(y == 0 & yhat == 1),  # FP
    sum(y == 1 & yhat == 0),  # FN
    sum(y == 1 & yhat == 1)   # TP
  )
}

## Compute the F1 score from the vector of cell probabilities
## p = (p00, p01, p10, p11) = (TN, FP, FN, TP) / N
f1_from_p <- function(p) {
  denom <- 2 * p[4] + p[2] + p[3]
  if (denom <= 0) return(NA_real_)
  2 * p[4] / denom
}

## Gradient of the F1 score with respect to p = (p00, p01, p10, p11)
grad_f1 <- function(p) {
  D <- 2 * p[4] + p[2] + p[3]
  if (D <= 0) return(rep(NA_real_, 4))
  c(
    0,
    -2 * p[4] / D^2,
    -2 * p[4] / D^2,
    2 * (p[2] + p[3]) / D^2
  )
}

## ---------------------------------------------------------
## 1. Cluster-robust covariance estimators
## ---------------------------------------------------------
sandwich_single <- function(cluster_counts, cluster_sizes) {
  N <- sum(cluster_sizes)
  p_hat <- colSums(cluster_counts) / N
  
  ## Plug-in centered cluster contributions:
  Uhat <- cluster_counts - cluster_sizes * matrix(
    p_hat,
    nrow = nrow(cluster_counts),
    ncol = 4,
    byrow = TRUE
  )
  
  ## Sandwich covariance estimator
  Omega_hat <- crossprod(Uhat) / N
  
  list(p_hat = p_hat, Omega_hat = Omega_hat)
}

## Cluster-robust covariance estimator for two paired models
sandwich_paired <- function(cluster_counts_c, cluster_counts_r, cluster_sizes) {
  N <- sum(cluster_sizes)
  
  p_hat_c <- colSums(cluster_counts_c) / N
  p_hat_r <- colSums(cluster_counts_r) / N
  
  Uhat_c <- cluster_counts_c - cluster_sizes * matrix(
    p_hat_c,
    nrow = nrow(cluster_counts_c),
    ncol = 4,
    byrow = TRUE
  )
  Uhat_r <- cluster_counts_r - cluster_sizes * matrix(
    p_hat_r,
    nrow = nrow(cluster_counts_r),
    ncol = 4,
    byrow = TRUE
  )
  
  Omega_c  <- crossprod(Uhat_c) / N
  Omega_r  <- crossprod(Uhat_r) / N
  Omega_cr <- crossprod(Uhat_c, Uhat_r) / N
  
  list(
    p_hat_c = p_hat_c,
    p_hat_r = p_hat_r,
    Omega_c = Omega_c,
    Omega_r = Omega_r,
    Omega_cr = Omega_cr
  )
}

## ---------------------------------------------------------
## 2. Data generation
## ---------------------------------------------------------
## Simulate clustered binary outcomes and paired model predictions.
##
## Data-generating mechanism:
##   - Each cluster has size m (simply fixed for demonstration purpose).
##   - A random effect b_i induces within-cluster dependence in the true label.
##   - Conditional on Y, each model predicts according to its sensitivity
##     and specificity parameters.
##
## Reference model (sensitivity, specificity): (se_ref, sp_ref)
## Candidate model (sensitivity, specificity): (se_cand, sp_cand)
simulate_clustered_binary <- function(
    n_clusters,
    m,
    gamma0 = -0.2,
    sigma_b = 0.8,
    se_ref = 0.82,
    sp_ref = 0.85,
    se_cand = 0.84,
    sp_cand = 0.84
) {
  ## Fixed cluster sizes throughout the simulation
  cluster_sizes <- rep(m, n_clusters)
  
  ## Each row stores the 4 confusion counts for one cluster
  cluster_counts_ref <- matrix(0, nrow = n_clusters, ncol = 4)
  cluster_counts_cand <- matrix(0, nrow = n_clusters, ncol = 4)
  
  for (i in seq_len(n_clusters)) {
    ## Cluster-specific random effect for the true label prevalence
    b_i <- rnorm(1, mean = 0, sd = sigma_b)
    pi_i <- expit(gamma0 + b_i)
    
    ## True labels in cluster i
    y <- rbinom(m, size = 1, prob = pi_i)  # correlated
    
    ## Reference model predictions:
    ## if Y=1, predict 1 with probability se_ref
    ## if Y=0, predict 1 with probability 1-sp_ref
    p_ref <- ifelse(y == 1, se_ref, 1 - sp_ref)
    yhat_ref <- rbinom(m, size = 1, prob = p_ref)
    
    ## Candidate model predictions:
    ## if Y=1, predict 1 with probability se_cand
    ## if Y=0, predict 1 with probability 1-sp_cand
    p_cand <- ifelse(y == 1, se_cand, 1 - sp_cand)
    yhat_cand <- rbinom(m, size = 1, prob = p_cand)
    
    ## Store cluster-level confusion counts for both models
    cluster_counts_ref[i, ] <- confusion_counts(y, yhat_ref)
    cluster_counts_cand[i, ] <- confusion_counts(y, yhat_cand)
  }
  
  list(
    cluster_sizes = cluster_sizes,
    counts_ref = cluster_counts_ref,
    counts_cand = cluster_counts_cand
  )
}

## ---------------------------------------------------------
## 3. Inference functions
## ---------------------------------------------------------
## Superiority test
superiority_test <- function(cluster_counts, cluster_sizes, theta0) {
  fit <- sandwich_single(cluster_counts, cluster_sizes)
  
  p_hat <- fit$p_hat
  Omega_hat <- fit$Omega_hat
  
  ## Estimated F1 score and its delta-method variance
  theta_hat <- f1_from_p(p_hat)
  grad_hat <- grad_f1(p_hat)
  V_hat <- as.numeric(t(grad_hat) %*% Omega_hat %*% grad_hat)
  
  ## Wald-type superiority statistic
  N <- sum(cluster_sizes)
  z_stat <- sqrt(N) * (theta_hat - theta0) / sqrt(V_hat)
  pval <- 1 - pnorm(z_stat)
  
  list(theta_hat = theta_hat, V_hat = V_hat, z = z_stat, pval = pval)
}

## Noninferiority test for paired candidate/reference models
noninferiority_test <- function(cluster_counts_c, cluster_counts_r, cluster_sizes, Delta) {
  fit <- sandwich_paired(cluster_counts_c, cluster_counts_r, cluster_sizes)
  
  p_hat_c <- fit$p_hat_c
  p_hat_r <- fit$p_hat_r
  
  ## Estimated model-specific F1 scores and their difference
  theta_hat_c <- f1_from_p(p_hat_c)
  theta_hat_r <- f1_from_p(p_hat_r)
  d_hat <- theta_hat_c - theta_hat_r
  
  ## Gradients for candidate and reference F1 scores
  g_c <- grad_f1(p_hat_c)
  g_r <- grad_f1(p_hat_r)
  
  ## Delta-method variance for the paired difference in F1 scores
  Vd_hat <- as.numeric(
    t(g_c) %*% fit$Omega_c %*% g_c +
      t(g_r) %*% fit$Omega_r %*% g_r -
      2 * t(g_c) %*% fit$Omega_cr %*% g_r
  )
  
  ## Wald-type noninferiority statistic
  N <- sum(cluster_sizes)
  z_stat <- sqrt(N) * (d_hat + Delta) / sqrt(Vd_hat)
  pval <- 1 - pnorm(z_stat)
  
  list(
    theta_hat_c = theta_hat_c,
    theta_hat_r = theta_hat_r,
    d_hat = d_hat,
    Vd_hat = Vd_hat,
    z = z_stat,
    pval = pval
  )
}

## ---------------------------------------------------------
## 4. Pilot-based sample size formulas
## ---------------------------------------------------------
## Required total number of observations for superiority testing
required_N_superiority <- function(theta1_hat, V_hat, theta0, alpha = 0.05, power = 0.80) {
  za <- qnorm(1 - alpha)
  zb <- qnorm(power)
  ((za + zb)^2 * V_hat) / (theta1_hat - theta0)^2
}

## Required total number of observations for noninferiority testing
required_N_noninferiority <- function(d1_hat, Vd_hat, Delta, alpha = 0.05, power = 0.80) {
  za <- qnorm(1 - alpha)
  zb <- qnorm(power)
  ((za + zb)^2 * Vd_hat) / (d1_hat + Delta)^2
}

## ---------------------------------------------------------
## 5. Empirical power evaluation
## ---------------------------------------------------------
## Empirical power for superiority testing at a fixed design size
empirical_power_superiority <- function(
    n_clusters,
    m,
    nsim = 1000,
    alpha = 0.05,
    theta0,
    gamma0 = -0.2,
    sigma_b = 0.8,
    se_ref = 0.82,
    sp_ref = 0.85
) {
  rej <- logical(nsim)
  
  for (s in seq_len(nsim)) {
    dat <- simulate_clustered_binary(
      n_clusters = n_clusters,
      m = m,
      gamma0 = gamma0,
      sigma_b = sigma_b,
      se_ref = se_ref,
      sp_ref = sp_ref,
      se_cand = se_ref,
      sp_cand = sp_ref
    )
    
    out <- superiority_test(dat$counts_ref, dat$cluster_sizes, theta0 = theta0)
    rej[s] <- (out$z > qnorm(1 - alpha))
  }
  
  mean(rej)
}

## Empirical power for noninferiority testing at a fixed design size
empirical_power_noninferiority <- function(
    n_clusters,
    m,
    nsim = 1000,
    alpha = 0.05,
    Delta,
    gamma0 = -0.2,
    sigma_b = 0.8,
    se_ref = 0.82,
    sp_ref = 0.85,
    se_cand = 0.84,
    sp_cand = 0.84
) {
  rej <- logical(nsim)
  
  for (s in seq_len(nsim)) {
    dat <- simulate_clustered_binary(
      n_clusters = n_clusters,
      m = m,
      gamma0 = gamma0,
      sigma_b = sigma_b,
      se_ref = se_ref,
      sp_ref = sp_ref,
      se_cand = se_cand,
      sp_cand = sp_cand
    )
    
    out <- noninferiority_test(
      dat$counts_cand,
      dat$counts_ref,
      dat$cluster_sizes,
      Delta = Delta
    )
    rej[s] <- (out$z > qnorm(1 - alpha))
  }
  
  mean(rej)
}

## ---------------------------------------------------------
## 6. Implementation
## ---------------------------------------------------------
## Significance level used in both the design formulas and testing
alpha <- 0.05

## Target power used to determine the required sample size
target_power <- 0.80

## Common cluster size used throughout the simulation
m <- 100

## ---------------------------------------------------------
## 6.A. Superiority calibration
## ---------------------------------------------------------
## Set seed so that the following pilot and simulation results are reproducible
set.seed(123)

## Generate a large pilot dataset under the superiority alternative
pilot_sup <- simulate_clustered_binary(
  n_clusters = 2000,
  m = m,
  gamma0 = -0.2,
  sigma_b = 0.8,
  se_ref = 0.82,
  sp_ref = 0.85,
  se_cand = 0.82,
  sp_cand = 0.85
)

## Estimate the pilot F1 score and its asymptotic variance
pilot_sup_fit <- superiority_test(
  cluster_counts = pilot_sup$counts_ref,
  cluster_sizes = pilot_sup$cluster_sizes,
  theta0 = 0.8
)

theta1_hat_sup <- pilot_sup_fit$theta_hat
V_hat_sup <- pilot_sup_fit$V_hat

## Plug the pilot estimates into the asymptotic sample size formula
N_req_sup <- required_N_superiority(
  theta1_hat = theta1_hat_sup,
  V_hat = V_hat_sup,
  theta0 = 0.8,
  alpha = alpha,
  power = target_power
)

## Convert total number of observations to number of clusters
n_req_sup <- ceiling(N_req_sup / m)

cat("Superiority pilot estimate:\n")
cat("theta1_hat =", round(theta1_hat_sup, 4), "\n")
cat("V_hat      =", round(V_hat_sup, 6), "\n")
cat("Required total N =", ceiling(N_req_sup), "\n")
cat("Required clusters =", n_req_sup, "\n\n")

## Evaluate empirical power at 80%, 100%, and 120% of the formula-based design
grid_sup <- c(
  ceiling(0.8 * n_req_sup),
  n_req_sup,
  ceiling(1.2 * n_req_sup)
)

power_sup <- sapply(grid_sup, function(nc) {
  empirical_power_superiority(
    n_clusters = nc,
    m = m,
    nsim = 1000,
    alpha = alpha,
    theta0 = 0.8,
    gamma0 = -0.2,
    sigma_b = 0.8,
    se_ref = 0.82,
    sp_ref = 0.85
  )
})

results_sup <- data.frame(
  setting = "Superiority",
  n_clusters = grid_sup,
  total_N = grid_sup * m,
  empirical_power = round(power_sup, 3)
)

print(results_sup)

## ---------------------------------------------------------
## 6.B. Noninferiority calibration
## ---------------------------------------------------------
## Reset the seed to reproduce the same pilot-based noninferiority result
set.seed(123)

## Generate a large pilot dataset under the noninferiority alternative
pilot_ni <- simulate_clustered_binary(
  n_clusters = 2000,
  m = m,
  gamma0 = -0.2,
  sigma_b = 0.8,
  se_ref = 0.82,
  sp_ref = 0.85,
  se_cand = 0.84,
  sp_cand = 0.84
)

## Estimate the pilot F1 difference and its asymptotic variance
pilot_ni_fit <- noninferiority_test(
  cluster_counts_c = pilot_ni$counts_cand,
  cluster_counts_r = pilot_ni$counts_ref,
  cluster_sizes = pilot_ni$cluster_sizes,
  Delta = 0.01
)

d1_hat_ni <- pilot_ni_fit$d_hat
Vd_hat_ni <- pilot_ni_fit$Vd_hat

## Plug the pilot estimates into the asymptotic noninferiority sample size formula
N_req_ni <- required_N_noninferiority(
  d1_hat = d1_hat_ni,
  Vd_hat = Vd_hat_ni,
  Delta = 0.01,
  alpha = alpha,
  power = target_power
)

## Convert total number of observations to number of clusters
n_req_ni <- ceiling(N_req_ni / m)

cat("\nNoninferiority pilot estimate:\n")
cat("d1_hat   =", round(d1_hat_ni, 4), "\n")
cat("Vd_hat   =", round(Vd_hat_ni, 6), "\n")
cat("Required total N =", ceiling(N_req_ni), "\n")
cat("Required clusters =", n_req_ni, "\n\n")

## Evaluate empirical power at 80%, 100%, and 120% of the formula-based design
grid_ni <- c(
  ceiling(0.8 * n_req_ni),
  n_req_ni,
  ceiling(1.2 * n_req_ni)
)

power_ni <- sapply(grid_ni, function(nc) {
  empirical_power_noninferiority(
    n_clusters = nc,
    m = m,
    nsim = 1000,
    alpha = alpha,
    Delta = 0.01,
    gamma0 = -0.2,
    sigma_b = 0.8,
    se_ref = 0.82,
    sp_ref = 0.85,
    se_cand = 0.84,
    sp_cand = 0.84
  )
})

results_ni <- data.frame(
  setting = "Noninferiority",
  n_clusters = grid_ni,
  total_N = grid_ni * m,
  empirical_power = round(power_ni, 3)
)

print(results_ni)
