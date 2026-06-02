rm(list = ls())

############################################################
##    Simulation code for finite performance evaluation   ##
############################################################
## 1. Respect manuscript assumptions                      ##
##    - independent clusters                              ##
##    - bounded cluster size                              ##
##    - stable target cell probabilities                  ##
##    - stable cluster-level covariance                   ##
## 2. Demonstrate finite-sample behavior of the proposed  ##
##    asymptotic inference framework                      ##
## 3. Compare cluster-robust SE/coverage with naive iid   ##
##    SE/coverage that ignores within-subject dependence  ##
############################################################

## ---------------------------------------------------------
## 0. Optional package check
## ---------------------------------------------------------
if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("Package 'MASS' is required. Please install it via install.packages('MASS').")
}

## ---------------------------------------------------------
## 1. Helper: build binary cell probabilities from
##    prevalence, sensitivity, specificity
## ---------------------------------------------------------
## p = (p11, p10, p01, p00) where
## p11 = P(pred = 1, true = 1) = TP probability
## p10 = P(pred = 1, true = 0) = FP probability
## p01 = P(pred = 0, true = 1) = FN probability
## p00 = P(pred = 0, true = 0) = TN probability
make_binary_prob <- function(prevalence, sensitivity, specificity) {
  p11 <- prevalence * sensitivity
  p01 <- prevalence * (1 - sensitivity)
  p00 <- (1 - prevalence) * specificity
  p10 <- (1 - prevalence) * (1 - specificity)
  
  p <- c(p11, p10, p01, p00)
  
  if (abs(sum(p) - 1) > 1e-10) {
    stop("Binary probability vector does not sum to 1.")
  }
  if (any(p <= 0)) {
    stop("Please choose strictly positive cell probabilities to avoid boundary issues.")
  }
  
  names(p) <- c("p11", "p10", "p01", "p00")
  return(p)
}

## ---------------------------------------------------------
## 1A. Helper: build a multiclass probability vector
##     from a confusion-probability matrix
## ---------------------------------------------------------
## Input:
##   P: an r x r matrix with rows = predicted class,
##      columns = true class
##
## Output:
##   A vector of length r^2 in row-major order:
##   (p11, p12, ..., p1r, p21, ..., prr)
make_multiclass_prob <- function(P) {
  if (!is.matrix(P)) stop("P must be a matrix.")
  if (nrow(P) != ncol(P)) stop("P must be a square matrix.")
  if (abs(sum(P) - 1) > 1e-10) stop("Multiclass confusion probability matrix must sum to 1.")
  if (any(P <= 0)) stop("All multiclass cell probabilities must be strictly positive.")
  
  p <- as.vector(t(P))
  r <- nrow(P)
  names(p) <- paste0("p", rep(seq_len(r), each = r), rep(seq_len(r), times = r))
  return(p)
}

## ---------------------------------------------------------
## 2. Correlation matrix generator
## ---------------------------------------------------------
## We impose within-subject dependence through a latent
## Gaussian copula. The observed category indicators are
## obtained by thresholding Gaussian latent variables.
##
## This gives:
## - exact marginal category probabilities
## - bounded cluster sizes
## - independent clusters
## - controllable within-subject dependence
##
## * The rho below is the latent Gaussian correlation, not
##   the exact observed indicator correlation.
make_latent_corr <- function(m, structure = c("cs", "ar1"), rho) {
  structure <- match.arg(structure)
  
  if (structure == "cs") {
    ## Compound symmetry:
    ## Corr(Z_j, Z_k) = rho for all j != k
    R <- matrix(rho, nrow = m, ncol = m)
    diag(R) <- 1
  } else {
    ## AR(1):
    ## Corr(Z_j, Z_k) = rho^{|j-k|}
    idx <- seq_len(m)
    R <- outer(idx, idx, function(a, b) rho^abs(a - b))
  }
  
  eigvals <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  if (min(eigvals) <= 0) {
    stop("The latent correlation matrix is not positive definite.")
  }
  
  return(R)
}

## ---------------------------------------------------------
## 3. Generic clustered categorical generator
## ---------------------------------------------------------
## Suppose there are K categories. In our single-label classification 
## with r classes, K = r^2.
##
## For each subject:
##   Z ~ MVN(0, R)
##   U = Phi(Z)
##   category = inverse-CDF(U; category probabilities)
simulate_cluster_categories <- function(prob_vec, m, corr_structure = "cs", rho) {
  K <- length(prob_vec)
  if (abs(sum(prob_vec) - 1) > 1e-10) stop("prob_vec must sum to 1.")
  if (any(prob_vec <= 0)) stop("All category probabilities must be > 0.")
  
  R <- make_latent_corr(m = m, structure = corr_structure, rho = rho)
  
  z <- MASS::mvrnorm(n = 1, mu = rep(0, m), Sigma = R)
  u <- pnorm(z)
  cuts <- cumsum(prob_vec)
  
  cat_id <- findInterval(u, vec = cuts) + 1
  cat_id[cat_id > K] <- K
  
  return(cat_id)
}

## ---------------------------------------------------------
## 4. Convert category labels to subject-level count vector
## ---------------------------------------------------------
category_counts <- function(cat_id, K) {
  tabulate(cat_id, nbins = K)
}

## ---------------------------------------------------------
## 5. Convert probability vector to confusion matrix
## ---------------------------------------------------------
## Probabilities are assumed to be ordered as
## (pred=1,true=1), (pred=1,true=2), ... , (pred=r,true=r)
prob_to_confmat <- function(prob_vec, r) {
  matrix(prob_vec, nrow = r, ncol = r, byrow = TRUE)
}

## ---------------------------------------------------------
## 6. Binary metric functions
## ---------------------------------------------------------
## Binary p = (p11, p10, p01, p00)
metric_precision_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]
  den <- p11 + p10
  if (den <= 0) return(NA_real_)
  p11 / den
}

metric_sensitivity_binary <- function(p) {
  p11 <- p[1]; p01 <- p[3]
  den <- p11 + p01
  if (den <= 0) return(NA_real_)
  p11 / den
}

metric_specificity_binary <- function(p) {
  p00 <- p[4]; p10 <- p[2]
  den <- p00 + p10
  if (den <= 0) return(NA_real_)
  p00 / den
}

metric_accuracy_binary <- function(p) {
  p[1] + p[4]
}

metric_f1_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]; p01 <- p[3]
  den <- 2 * p11 + p10 + p01
  if (den <= 0) return(NA_real_)
  2 * p11 / den
}

metric_mcc_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]; p01 <- p[3]; p00 <- p[4]
  den <- (p11 + p10) * (p11 + p01) * (p00 + p10) * (p00 + p01)
  if (den <= 0) return(NA_real_)
  (p11 * p00 - p10 * p01) / sqrt(den)
}

## ---------------------------------------------------------
## 6A. Multiclass metric functions
## ---------------------------------------------------------
metric_micro_f1_multiclass <- function(p, r) {
  P <- prob_to_confmat(p, r)
  sum(diag(P))
}

metric_macro_f1_multiclass <- function(p, r) {
  P <- prob_to_confmat(p, r)
  row_marg <- rowSums(P)
  col_marg <- colSums(P)
  
  class_f1 <- numeric(r)
  for (a in seq_len(r)) {
    den <- row_marg[a] + col_marg[a]
    if (den <= 0) return(NA_real_)
    class_f1[a] <- 2 * P[a, a] / den
  }
  mean(class_f1)
}

## ---------------------------------------------------------
## 7. Analytic gradients for binary metrics
## ---------------------------------------------------------
grad_precision_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]
  D <- p11 + p10
  if (D <= 0) return(rep(NA_real_, 4))
  
  c(
    p10 / D^2,
    -p11 / D^2,
    0,
    0
  )
}

grad_sensitivity_binary <- function(p) {
  p11 <- p[1]; p01 <- p[3]
  D <- p11 + p01
  if (D <= 0) return(rep(NA_real_, 4))
  
  c(
    p01 / D^2,
    0,
    -p11 / D^2,
    0
  )
}

grad_specificity_binary <- function(p) {
  p00 <- p[4]; p10 <- p[2]
  D <- p00 + p10
  if (D <= 0) return(rep(NA_real_, 4))
  
  c(
    0,
    -p00 / D^2,
    0,
    p10 / D^2
  )
}

grad_accuracy_binary <- function(p) {
  c(1, 0, 0, 1)
}

grad_f1_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]; p01 <- p[3]
  D <- 2 * p11 + p10 + p01
  if (D <= 0) return(rep(NA_real_, 4))
  
  c(
    2 * (p10 + p01) / D^2,
    -2 * p11 / D^2,
    -2 * p11 / D^2,
    0
  )
}

grad_mcc_binary <- function(p) {
  p11 <- p[1]; p10 <- p[2]; p01 <- p[3]; p00 <- p[4]
  
  A <- p11 * p00 - p10 * p01
  B <- (p11 + p10) * (p11 + p01) * (p00 + p10) * (p00 + p01)
  
  if (B <= 0) return(rep(NA_real_, 4))
  D <- sqrt(B)
  
  gradA <- c(p00, -p01, -p10, p11)
  
  dB_dp11 <- (p11 + p01) * (p00 + p10) * (p00 + p01) +
    (p11 + p10) * (p00 + p10) * (p00 + p01)
  
  dB_dp10 <- (p11 + p01) * (p00 + p10) * (p00 + p01) +
    (p11 + p10) * (p11 + p01) * (p00 + p01)
  
  dB_dp01 <- (p11 + p10) * (p00 + p10) * (p00 + p01) +
    (p11 + p10) * (p11 + p01) * (p00 + p10)
  
  dB_dp00 <- (p11 + p10) * (p11 + p01) * (p00 + p01) +
    (p11 + p10) * (p11 + p01) * (p00 + p10)
  
  gradB <- c(dB_dp11, dB_dp10, dB_dp01, dB_dp00)
  
  gradA / D - (A / (2 * D^3)) * gradB
}

## ---------------------------------------------------------
## 8. Generic finite-difference gradient
## ---------------------------------------------------------
## Used for multiclass metrics where analytic gradients are
## not explicitly coded.
fd_gradient <- function(fun, p, eps = 1e-7, ...) {
  g <- numeric(length(p))
  for (k in seq_along(p)) {
    p_plus  <- p
    p_minus <- p
    p_plus[k]  <- p_plus[k] + eps
    p_minus[k] <- p_minus[k] - eps
    g[k] <- (fun(p_plus, ...) - fun(p_minus, ...)) / (2 * eps)
  }
  return(g)
}

## ---------------------------------------------------------
## 9. Metric registries
## ---------------------------------------------------------
## Registry-driven simulation:
## each metric has
## - "fun": metric function
## - "grad": analytic gradient if available
get_metric_registry_binary <- function() {
  list(
    Precision = list(fun = metric_precision_binary, grad = grad_precision_binary),
    Sensitivity = list(fun = metric_sensitivity_binary, grad = grad_sensitivity_binary),
    Specificity = list(fun = metric_specificity_binary, grad = grad_specificity_binary),
    Accuracy = list(fun = metric_accuracy_binary, grad = grad_accuracy_binary),
    F1 = list(fun = metric_f1_binary, grad = grad_f1_binary),
    MCC = list(fun = metric_mcc_binary, grad = grad_mcc_binary)
  )
}

get_metric_registry_multiclass <- function(r) {
  list(
    microF1 = list(
      fun = function(p) metric_micro_f1_multiclass(p, r = r),
      grad = NULL
    ),
    macroF1 = list(
      fun = function(p) metric_macro_f1_multiclass(p, r = r),
      grad = NULL
    )
  )
}

## ---------------------------------------------------------
## 10. Simulate one clustered dataset
## ---------------------------------------------------------
simulate_dataset_clustered <- function(n_subjects,
                                       prob_vec,
                                       cluster_size_values = 5:10,
                                       corr_structure = "cs",
                                       rho) {
  K <- length(prob_vec)
  
  ## Draw cluster sizes independently from the allowed set
  m_i <- sample(cluster_size_values, size = n_subjects, replace = TRUE)
  
  ## Subject-level count matrix
  S_mat <- matrix(0, nrow = n_subjects, ncol = K)
  
  for (i in seq_len(n_subjects)) {
    cats_i <- simulate_cluster_categories(
      prob_vec = prob_vec,
      m = m_i[i],
      corr_structure = corr_structure,
      rho = rho
    )
    S_mat[i, ] <- category_counts(cats_i, K = K)
  }
  
  N_total <- sum(m_i)
  p_hat <- colSums(S_mat) / N_total
  
  list(
    S_mat = S_mat,
    m_i = m_i,
    N = N_total,
    p_hat = p_hat
  )
}

## ---------------------------------------------------------
## 11. Cluster-robust sandwich covariance for p-hat
## ---------------------------------------------------------
sandwich_Omega_hat <- function(S_mat, m_i, p_hat) {
  n <- nrow(S_mat)
  K <- ncol(S_mat)
  
  U_mat <- matrix(0, nrow = n, ncol = K)
  for (i in seq_len(n)) {
    U_mat[i, ] <- S_mat[i, ] - m_i[i] * p_hat
  }
  
  Omega_hat <- crossprod(U_mat) / sum(m_i)
  list(Omega_hat = Omega_hat, U_mat = U_mat)
}

## ---------------------------------------------------------
## 12. Naive iid covariance for p-hat
## ---------------------------------------------------------
## This ignores clustering and treats each lower-level
## observation as independent multinomial.
##
## For one observation with category probability vector p:
## Var(Z) = diag(p) - p p^T
naive_Sigma_hat <- function(p_hat) {
  diag(p_hat) - tcrossprod(p_hat)
}

## ---------------------------------------------------------
## 13. Asymptotic SE for one metric:
##     cluster-robust and naive
## ---------------------------------------------------------
asymptotic_se_metric <- function(p_hat, S_mat, m_i, metric_fun, grad_fun = NULL) {
  sand <- sandwich_Omega_hat(S_mat = S_mat, m_i = m_i, p_hat = p_hat)
  Omega_hat <- sand$Omega_hat
  
  if (is.null(grad_fun)) {
    grad_hat <- fd_gradient(metric_fun, p_hat)
  } else {
    grad_hat <- grad_fun(p_hat)
  }
  
  V_hat <- as.numeric(t(grad_hat) %*% Omega_hat %*% grad_hat)
  se_hat <- sqrt(V_hat / sum(m_i))
  
  list(se = se_hat, V = V_hat, grad = grad_hat, Omega_hat = Omega_hat)
}

naive_se_metric <- function(p_hat, metric_fun, grad_fun = NULL, N_total) {
  Sigma_hat <- naive_Sigma_hat(p_hat)
  
  if (is.null(grad_fun)) {
    grad_hat <- fd_gradient(metric_fun, p_hat)
  } else {
    grad_hat <- grad_fun(p_hat)
  }
  
  V_hat_naive <- as.numeric(t(grad_hat) %*% Sigma_hat %*% grad_hat)
  se_hat_naive <- sqrt(V_hat_naive / N_total)
  
  list(se = se_hat_naive, V = V_hat_naive, grad = grad_hat, Sigma_hat = Sigma_hat)
}

## ---------------------------------------------------------
## 14. One Monte Carlo replicate
## ---------------------------------------------------------
## For a single simulated dataset:
## - compute theta_hat for each metric
## - cluster-robust asymptotic SE and Wald CI
## - naive iid asymptotic SE and Wald CI
one_replication <- function(n_subjects,
                            prob_vec,
                            metrics,
                            metric_registry,
                            cluster_size_values = 5:10,
                            corr_structure = "cs",
                            rho,
                            ci_level = 0.95) {
  
  reg <- metric_registry
  
  dat <- simulate_dataset_clustered(
    n_subjects = n_subjects,
    prob_vec = prob_vec,
    cluster_size_values = cluster_size_values,
    corr_structure = corr_structure,
    rho = rho
  )
  
  S_mat <- dat$S_mat
  m_i   <- dat$m_i
  p_hat <- dat$p_hat
  N     <- dat$N
  
  alpha <- 1 - ci_level
  zcrit <- qnorm(1 - alpha / 2)
  
  out_list <- vector("list", length(metrics))
  
  for (k in seq_along(metrics)) {
    metric_name <- metrics[k]
    metric_fun  <- reg[[metric_name]]$fun
    grad_fun    <- reg[[metric_name]]$grad
    
    theta_true <- metric_fun(prob_vec)
    theta_hat  <- metric_fun(p_hat)
    
    ## Cluster-robust asymptotic SE
    asy <- asymptotic_se_metric(
      p_hat = p_hat,
      S_mat = S_mat,
      m_i = m_i,
      metric_fun = metric_fun,
      grad_fun = grad_fun
    )
    
    se_asy <- asy$se
    ci_asy <- c(theta_hat - zcrit * se_asy,
                theta_hat + zcrit * se_asy)
    
    ## Naive iid SE
    nai <- naive_se_metric(
      p_hat = p_hat,
      metric_fun = metric_fun,
      grad_fun = grad_fun,
      N_total = N
    )
    
    se_naive <- nai$se
    ci_naive <- c(theta_hat - zcrit * se_naive,
                  theta_hat + zcrit * se_naive)
    
    out_list[[k]] <- data.frame(
      metric = metric_name,
      theta_true = theta_true,
      theta_hat = theta_hat,
      bias = theta_hat - theta_true,
      se_asy = se_asy,
      se_naive = se_naive,
      cover_asy = as.integer(ci_asy[1] <= theta_true && theta_true <= ci_asy[2]),
      cover_naive = as.integer(ci_naive[1] <= theta_true && theta_true <= ci_naive[2]),
      N_total = N
    )
  }
  
  out <- do.call(rbind, out_list)
  return(out)
}

## ---------------------------------------------------------
## 15. Run all replications for one design point
## ---------------------------------------------------------
run_design_point <- function(nsim,
                             n_subjects,
                             prob_vec,
                             scenario_name,
                             metrics,
                             metric_registry,
                             cluster_size_values,
                             corr_structure,
                             rho,
                             ci_level = 0.95,
                             progress = TRUE) {
  res <- vector("list", nsim)
  
  if (progress) {
    message("Running scenario = ", scenario_name,
            ", n = ", n_subjects,
            ", corr = ", corr_structure,
            ", rho = ", rho)
  }
  
  for (s in seq_len(nsim)) {
    if (progress && (s %% 50 == 0)) {
      message("  replication ", s, " / ", nsim)
    }
    
    tmp <- one_replication(
      n_subjects = n_subjects,
      prob_vec = prob_vec,
      metrics = metrics,
      metric_registry = metric_registry,
      cluster_size_values = cluster_size_values,
      corr_structure = corr_structure,
      rho = rho,
      ci_level = ci_level
    )
    
    tmp$scenario <- scenario_name
    tmp$n_subjects <- n_subjects
    tmp$corr_structure <- corr_structure
    tmp$rho <- rho
    tmp$rep <- s
    
    res[[s]] <- tmp
  }
  
  out <- do.call(rbind, res)
  rownames(out) <- NULL
  return(out)
}

## ---------------------------------------------------------
## 16. Summarize Monte Carlo results
## ---------------------------------------------------------
## For theorem validation, the most important columns are:
## - mean(theta_hat) vs theta_true  -> bias
## - sd(theta_hat)                  -> empirical SE (ESE)
## - mean(se_asy)                   -> average robust ASE
## - mean(se_naive)                 -> average naive ASE
## - coverage_asymptotic            -> robust Wald CP
## - coverage_naive                 -> naive Wald CP
summarize_simulation <- function(raw_res) {
  split_key <- interaction(raw_res$scenario,
                           raw_res$n_subjects,
                           raw_res$corr_structure,
                           raw_res$rho,
                           raw_res$metric,
                           drop = TRUE)
  
  chunks <- split(raw_res, split_key)
  
  out <- lapply(chunks, function(df) {
    data.frame(
      scenario = df$scenario[1],
      n_subjects = df$n_subjects[1],
      corr_structure = df$corr_structure[1],
      rho = df$rho[1],
      metric = df$metric[1],
      theta_true = df$theta_true[1],
      mean_theta_hat = mean(df$theta_hat, na.rm = TRUE),
      bias = mean(df$bias, na.rm = TRUE),
      empirical_se = sd(df$theta_hat, na.rm = TRUE),
      mean_asymptotic_se = mean(df$se_asy, na.rm = TRUE),
      mean_naive_se = mean(df$se_naive, na.rm = TRUE),
      coverage_asymptotic = mean(df$cover_asy, na.rm = TRUE),
      coverage_naive = mean(df$cover_naive, na.rm = TRUE),
      mean_total_observations = mean(df$N_total, na.rm = TRUE)
    )
  })
  
  out <- do.call(rbind, out)
  rownames(out) <- NULL
  return(out)
}

## ---------------------------------------------------------
## 17. Default binary scenarios
## ---------------------------------------------------------
## Scenario 1: balanced / moderate
##   prevalence = 0.50, sensitivity = 0.70, specificity = 0.70
##
## Scenario 2: imbalanced / stronger classifier
##   prevalence = 0.20, sensitivity = 0.80, specificity = 0.90
default_scenarios <- list(
  balanced_moderate = make_binary_prob(prevalence = 0.50, sensitivity = 0.70, specificity = 0.70),
  imbalanced_strong = make_binary_prob(prevalence = 0.20, sensitivity = 0.80, specificity = 0.90)
)

## ---------------------------------------------------------
## 17A. Example multiclass scenarios
## ---------------------------------------------------------
## We choose strictly positive probability matrices so that
## target metrics lie in the interior of the smoothness domain.
default_multiclass_scenarios <- list(
  mc3_balanced = make_multiclass_prob(
    matrix(c(
      0.25, 0.03, 0.02,
      0.04, 0.30, 0.03,
      0.02, 0.04, 0.27
    ), nrow = 3, byrow = TRUE)
  ),
  mc3_imbalanced = make_multiclass_prob(
    matrix(c(
      0.42, 0.03, 0.02,
      0.05, 0.18, 0.03,
      0.04, 0.05, 0.18
    ), nrow = 3, byrow = TRUE)
  )
)

## ---------------------------------------------------------
## 18. High-level wrapper for the full simulation grid
## ---------------------------------------------------------
run_full_simulation <- function(nsim = 500,
                                n_subjects_grid = c(50, 100, 200),
                                scenarios,
                                metrics,
                                metric_registry,
                                cluster_size_values,
                                corr_structures = c("cs", "ar1"),
                                rho_values,
                                ci_level = 0.95,
                                progress = TRUE) {
  all_res <- list()
  counter <- 1L
  
  for (sc_name in names(scenarios)) {
    p_true <- scenarios[[sc_name]]
    
    for (n_sub in n_subjects_grid) {
      for (corr_str in corr_structures) {
        for (rho in rho_values) {
          tmp <- run_design_point(
            nsim = nsim,
            n_subjects = n_sub,
            prob_vec = p_true,
            scenario_name = sc_name,
            metrics = metrics,
            metric_registry = metric_registry,
            cluster_size_values = cluster_size_values,
            corr_structure = corr_str,
            rho = rho,
            ci_level = ci_level,
            progress = progress
          )
          
          all_res[[counter]] <- tmp
          counter <- counter + 1L
        }
      }
    }
  }
  
  raw_res <- do.call(rbind, all_res)
  rownames(raw_res) <- NULL
  
  summary_res <- summarize_simulation(raw_res)
  
  list(raw = raw_res, summary = summary_res)
}

## ---------------------------------------------------------
## 19. Example run: binary
## ---------------------------------------------------------
set.seed(10903)

binary_registry <- get_metric_registry_binary()

sim_out <- run_full_simulation(
  nsim = 2000,
  n_subjects_grid = c(50, 100, 200),
  scenarios = default_scenarios,
  metrics = c("Precision", "Sensitivity", "Specificity", "Accuracy", "F1", "MCC"),
  metric_registry = binary_registry,
  cluster_size_values = 100:300,
  corr_structures = c("cs", "ar1"),
  rho_values = c(0.5,0.8),
  ci_level = 0.95,
  progress = TRUE
)

## ---------------------------------------------------------
## 20. Main summary table: binary
## ---------------------------------------------------------
sim_out$summary <- sim_out$summary[
  order(sim_out$summary$scenario,
        sim_out$summary$metric,
        sim_out$summary$n_subjects,
        sim_out$summary$corr_structure),
]


## ---------------------------------------------------------
## 21. Optional: write binary results to CSV
## ---------------------------------------------------------
## write.csv(sim_out$summary, "cluster_metric_sim_summary_binary.csv", row.names = FALSE)
## write.csv(sim_out$raw,     "cluster_metric_sim_raw_binary.csv", row.names = FALSE)

## ---------------------------------------------------------
## 22. Example run: multiclass
## ---------------------------------------------------------
## Same engine, now with:
## - length(prob_vec) = r^2
## - metrics = c("microF1", "macroF1")
set.seed(10903)

multiclass_registry <- get_metric_registry_multiclass(r = 3)

sim_out_mc <- run_full_simulation(
  nsim = 2000,
  n_subjects_grid = c(50, 100, 200),
  scenarios = default_multiclass_scenarios,
  metrics = c("microF1", "macroF1"),
  metric_registry = multiclass_registry,
  cluster_size_values = 100:300,
  corr_structures = c("cs", "ar1"),
  rho_values = c(0.5,0.8),
  ci_level = 0.95,
  progress = TRUE
)

## ---------------------------------------------------------
## 23. Main summary table: multiclass
## ---------------------------------------------------------
sim_out_mc$summary <- sim_out_mc$summary[
  order(sim_out_mc$summary$scenario,
        sim_out_mc$summary$metric,
        sim_out_mc$summary$n_subjects,
        sim_out_mc$summary$corr_structure),
]

## For binary summary table: multiply columns 6:end by 100, then round to 1 decimal
sim_out$summary[, 6:(ncol(sim_out$summary)-1)] <-
  round(sim_out$summary[, 6:(ncol(sim_out$summary)-1)] * 100, 1)

## For multiclass summary table: multiply columns 6:end by 100, then round to 1 decimal
sim_out_mc$summary[, 6:(ncol(sim_out_mc$summary)-1)] <-
  round(sim_out_mc$summary[, 6:(ncol(sim_out_mc$summary)-1)] * 100, 1)

print(sim_out)
print(sim_out_mc)

## ---------------------------------------------------------
## 24. Optional: write multiclass results to CSV
## ---------------------------------------------------------
## write.csv(sim_out_mc$summary, "cluster_metric_sim_summary_multiclass.csv", row.names = FALSE)
## write.csv(sim_out_mc$raw,     "cluster_metric_sim_raw_multiclass.csv", row.names = FALSE)
