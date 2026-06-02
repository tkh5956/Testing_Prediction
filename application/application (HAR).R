rm(list = ls())

############################################################
##          Application code for UCI HAR Dataset          ##
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
## 0. Setup
## ---------------------------------------------------------
## Install packages 
packages <- c("dplyr", "data.table", "randomForest", "numDeriv", "ggplot2","patchwork")
new_pkgs <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(dplyr)
library(data.table)
library(randomForest)
library(numDeriv)
library(ggplot2)
library(patchwork)
set.seed(10903)

## Download and unzip HAR dataset 
## (http://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

data_path <- file.path("YOUR_PATH_to_UCI_HAR_Dataset")

## Load data: Training set
X_train <- read.table(file.path(data_path, "train", "X_train.txt"))
y_train <- read.table(file.path(data_path, "train", "y_train.txt"))
subject_train <- read.table(file.path(data_path, "train", "subject_train.txt"))

## Load data: Test set
X_test <- read.table(file.path(data_path, "test", "X_test.txt"))
y_test <- read.table(file.path(data_path, "test", "y_test.txt"))
subject_test <- read.table(file.path(data_path, "test", "subject_test.txt"))

## Combined dataset
X_all <- rbind(X_train, X_test)
y_all <- rbind(y_train, y_test)
subject_all <- rbind(subject_train, subject_test)

colnames(y_all) <- "y_true"
colnames(subject_all) <- "subject_id"

har_df <- data.frame(
  subject_id = subject_all$subject_id,
  y_true = y_all$y_true
)

## Within-subject index
har_df <- har_df %>%
  group_by(subject_id) %>%
  mutate(obs_index = row_number()) %>%
  ungroup()

## ---------------------------------------------------------
## 1. Split 30 subjects into:
##    (1) Stage1_training: 2 subjects
##    (2) Stage1_testing : 3 subjects
##    (3) Stage2_testing : 25 subjects
## Note: This partition is for demonstrative purpose ONLY
## ---------------------------------------------------------
subjects <- sort(unique(har_df$subject_id))

Stage1_train_subjects <- sample(subjects, 2)
remaining_subjects <- setdiff(subjects, Stage1_train_subjects)
Stage1_test_subjects <- sample(remaining_subjects, 3)
Stage2_test_subjects <- setdiff(remaining_subjects, Stage1_test_subjects)

cat("Stage 1 training subjects:", Stage1_train_subjects, "\n")
cat("Stage 1 testing subjects :", Stage1_test_subjects, "\n")
cat("Stage 2 testing subjects :", Stage2_test_subjects, "\n")

idx_Stage1_train <- which(har_df$subject_id %in% Stage1_train_subjects)
idx_Stage1_test  <- which(har_df$subject_id %in% Stage1_test_subjects)
idx_Stage2_test  <- which(har_df$subject_id %in% Stage2_test_subjects)

X_Stage1_train <- X_all[idx_Stage1_train, ]
X_Stage1_test  <- X_all[idx_Stage1_test, ]
X_Stage2_test  <- X_all[idx_Stage2_test, ]

y_Stage1_train <- as.factor(har_df$y_true[idx_Stage1_train])
y_Stage1_test  <- as.factor(har_df$y_true[idx_Stage1_test])
y_Stage2_test  <- as.factor(har_df$y_true[idx_Stage2_test])

df_Stage1_test <- har_df[idx_Stage1_test, c("subject_id", "obs_index", "y_true")]
df_Stage2_test <- har_df[idx_Stage2_test, c("subject_id", "obs_index", "y_true")]

## ---------------------------------------------------------
## 2. Fit two models on Stage1_training
##    - Gold standard (reference) model: rf_ref (ntree = 100)
##    - Proposed candidate model: rf_can (ntree = 10)
## ---------------------------------------------------------
set.seed(1234)
rf_ref <- randomForest(
  x = X_Stage1_train,
  y = y_Stage1_train,
  ntree = 100
)

rf_cand <- randomForest(
  x = X_Stage1_train,
  y = y_Stage1_train,
  ntree = 10
)

## ---------------------------------------------------------
## 3. Obtain predictions on Stage1_testing and Stage2_testing
## ---------------------------------------------------------
pred_ref_Stage1  <- predict(rf_ref,  X_Stage1_test)
pred_cand_Stage1 <- predict(rf_cand, X_Stage1_test)

pred_ref_Stage2  <- predict(rf_ref,  X_Stage2_test)
pred_cand_Stage2 <- predict(rf_cand, X_Stage2_test)

## ---------------------------------------------------------
## 4. Build analysis datasets
## ---------------------------------------------------------
## Stage 1 testing: single-model datasets
Stage1_ref_data <- data.frame(
  subject_id = df_Stage1_test$subject_id,
  obs_index  = df_Stage1_test$obs_index,
  y_true     = as.integer(df_Stage1_test$y_true),
  y_pred     = as.integer(pred_ref_Stage1)
)

Stage1_cand_data <- data.frame(
  subject_id = df_Stage1_test$subject_id,
  obs_index  = df_Stage1_test$obs_index,
  y_true     = as.integer(df_Stage1_test$y_true),
  y_pred     = as.integer(pred_cand_Stage1)
)

## Stage 1 testing: paired-model dataset
Stage1_pair_data <- data.frame(
  subject_id   = df_Stage1_test$subject_id,
  obs_index    = df_Stage1_test$obs_index,
  y_true       = as.integer(df_Stage1_test$y_true),
  y_pred_ref   = as.integer(pred_ref_Stage1),
  y_pred_cand  = as.integer(pred_cand_Stage1)
)

## Stage 2 testing: single-model datasets
Stage2_ref_data <- data.frame(
  subject_id = df_Stage2_test$subject_id,
  obs_index  = df_Stage2_test$obs_index,
  y_true     = as.integer(df_Stage2_test$y_true),
  y_pred     = as.integer(pred_ref_Stage2)
)

Stage2_cand_data <- data.frame(
  subject_id = df_Stage2_test$subject_id,
  obs_index  = df_Stage2_test$obs_index,
  y_true     = as.integer(df_Stage2_test$y_true),
  y_pred     = as.integer(pred_cand_Stage2)
)

## Stage 2 testing: paired-model dataset
Stage2_pair_data <- data.frame(
  subject_id   = df_Stage2_test$subject_id,
  obs_index    = df_Stage2_test$obs_index,
  y_true       = as.integer(df_Stage2_test$y_true),
  y_pred_ref   = as.integer(pred_ref_Stage2),
  y_pred_cand  = as.integer(pred_cand_Stage2)
)

## ---------------------------------------------------------
## 5. Utility functions
## ---------------------------------------------------------
make_confusion_grid <- function(classes) {
  expand.grid(pred = classes, truth = classes,
              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
}

vec_to_mat <- function(p, classes) {
  r <- length(classes)
  matrix(p, nrow = r, ncol = r, byrow = TRUE,
         dimnames = list(pred = classes, truth = classes))
}

stabilize_prob <- function(p, eps = 1e-10) {
  p <- pmax(p, eps)
  p / sum(p)
}

metric_micro_f1 <- function(p, classes) {
  P <- vec_to_mat(stabilize_prob(p), classes)
  sum(diag(P))
}

metric_macro_f1 <- function(p, classes) {
  P <- vec_to_mat(stabilize_prob(p), classes)
  row_marg <- rowSums(P)
  col_marg <- colSums(P)
  f1_per_class <- 2 * diag(P) / (row_marg + col_marg)
  mean(f1_per_class)
}

metric_value <- function(p, classes, metric = c("micro_f1", "macro_f1")) {
  metric <- match.arg(metric)
  if (metric == "micro_f1") return(metric_micro_f1(p, classes))
  if (metric == "macro_f1") return(metric_macro_f1(p, classes))
}

## ---------------------------------------------------------
## 6. Estimate pilot quantities
## ---------------------------------------------------------
## For single model
estimate_single_model <- function(data_4col,
                                  classes = sort(unique(c(data_4col$y_true,
                                                          data_4col$y_pred))),
                                  metric = c("micro_f1", "macro_f1")) {
  metric <- match.arg(metric)
  classes <- sort(classes)
  r <- length(classes)
  grid <- make_confusion_grid(classes)
  
  dat <- data_4col %>%
    mutate(
      y_true = factor(y_true, levels = classes),
      y_pred = factor(y_pred, levels = classes)
    )
  
  N <- nrow(dat)
  
  get_onehot <- function(pred_val, truth_val) {
    as.numeric(grid$pred == pred_val & grid$truth == truth_val)
  }
  
  Z_mat <- t(mapply(get_onehot,
                    pred_val = as.character(dat$y_pred),
                    truth_val = as.character(dat$y_true)))
  
  p_hat <- colMeans(Z_mat)
  
  subject_ids <- unique(as.character(dat$subject_id))
  S_list <- lapply(subject_ids, function(id) {
    rows <- which(dat$subject_id == id)
    colSums(Z_mat[rows, , drop = FALSE])
  })
  S_mat <- do.call(rbind, S_list)
  m_vec <- rowSums(S_mat)
  
  U_mat <- S_mat - m_vec %o% p_hat
  
  Omega_hat <- matrix(0, nrow = r^2, ncol = r^2)
  for (i in seq_len(nrow(U_mat))) {
    Omega_hat <- Omega_hat + tcrossprod(U_mat[i, ])
  }
  Omega_hat <- Omega_hat / N
  
  g_fun <- function(p) metric_value(p, classes = classes, metric = metric)
  grad_hat <- grad(func = g_fun, x = p_hat)
  
  theta_hat <- g_fun(p_hat)
  V_hat <- as.numeric(t(grad_hat) %*% Omega_hat %*% grad_hat)
  mbar_hat <- mean(m_vec)
  
  se_hat <- sqrt(V_hat / N)
  
  list(
    classes = classes,
    metric = metric,
    N = N,
    n_subjects = length(subject_ids),
    mbar_hat = mbar_hat,
    p_hat = p_hat,
    theta_hat = theta_hat,
    Omega_hat = Omega_hat,
    grad_hat = grad_hat,
    V_hat = V_hat,
    se_hat = se_hat
  )
}

## For paired models
estimate_paired_models <- function(data_pair,
                                   classes = sort(unique(c(data_pair$y_true,
                                                           data_pair$y_pred_ref,
                                                           data_pair$y_pred_cand))),
                                   metric = c("micro_f1", "macro_f1")) {
  metric <- match.arg(metric)
  classes <- sort(classes)
  r <- length(classes)
  
  grid3 <- expand.grid(
    pred_cand = classes,
    pred_ref  = classes,
    truth     = classes,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  dat <- data_pair %>%
    mutate(
      y_true = factor(y_true, levels = classes),
      y_pred_ref = factor(y_pred_ref, levels = classes),
      y_pred_cand = factor(y_pred_cand, levels = classes)
    )
  
  N <- nrow(dat)
  
  get_onehot3 <- function(pc, pr, tr) {
    as.numeric(grid3$pred_cand == pc &
                 grid3$pred_ref  == pr &
                 grid3$truth     == tr)
  }
  
  W_mat <- t(mapply(get_onehot3,
                    pc = as.character(dat$y_pred_cand),
                    pr = as.character(dat$y_pred_ref),
                    tr = as.character(dat$y_true)))
  
  q_hat <- colMeans(W_mat)
  
  q_to_p_cand <- function(q) {
    p <- numeric(r^2)
    grid2 <- make_confusion_grid(classes)
    for (k in seq_len(nrow(grid2))) {
      a <- grid2$pred[k]
      b <- grid2$truth[k]
      idx <- which(grid3$pred_cand == a & grid3$truth == b)
      p[k] <- sum(q[idx])
    }
    p
  }
  
  q_to_p_ref <- function(q) {
    p <- numeric(r^2)
    grid2 <- make_confusion_grid(classes)
    for (k in seq_len(nrow(grid2))) {
      a <- grid2$pred[k]
      b <- grid2$truth[k]
      idx <- which(grid3$pred_ref == a & grid3$truth == b)
      p[k] <- sum(q[idx])
    }
    p
  }
  
  h_fun <- function(q) {
    p_c <- q_to_p_cand(q)
    p_r <- q_to_p_ref(q)
    metric_value(p_c, classes = classes, metric = metric) -
      metric_value(p_r, classes = classes, metric = metric)
  }
  
  d_hat <- h_fun(q_hat)
  
  subject_ids <- unique(as.character(dat$subject_id))
  T_list <- lapply(subject_ids, function(id) {
    rows <- which(dat$subject_id == id)
    colSums(W_mat[rows, , drop = FALSE])
  })
  T_mat <- do.call(rbind, T_list)
  m_vec <- rowSums(T_mat)
  
  Vcluster_mat <- T_mat - m_vec %o% q_hat
  
  Xi_hat <- matrix(0, nrow = r^3, ncol = r^3)
  for (i in seq_len(nrow(Vcluster_mat))) {
    Xi_hat <- Xi_hat + tcrossprod(Vcluster_mat[i, ])
  }
  Xi_hat <- Xi_hat / N
  
  grad_hat <- grad(func = h_fun, x = q_hat)
  Vd_hat <- as.numeric(t(grad_hat) %*% Xi_hat %*% grad_hat)
  mbar_hat <- mean(m_vec)
  
  se_d_hat <- sqrt(Vd_hat / N)
  
  list(
    classes = classes,
    metric = metric,
    N = N,
    n_subjects = length(subject_ids),
    mbar_hat = mbar_hat,
    q_hat = q_hat,
    d_hat = d_hat,
    Xi_hat = Xi_hat,
    grad_hat = grad_hat,
    Vd_hat = Vd_hat,
    se_d_hat = se_d_hat
  )
}

## ---------------------------------------------------------
## 7. Naive variance estimators (ignoring clustering)
## ---------------------------------------------------------
## For single model
estimate_single_model_naive <- function(data_4col,
                                        classes = sort(unique(c(data_4col$y_true,
                                                                data_4col$y_pred))),
                                        metric = c("micro_f1", "macro_f1")) {
  metric <- match.arg(metric)
  classes <- sort(classes)
  r <- length(classes)
  grid <- make_confusion_grid(classes)
  
  dat <- data_4col %>%
    mutate(
      y_true = factor(y_true, levels = classes),
      y_pred = factor(y_pred, levels = classes)
    )
  
  N <- nrow(dat)
  
  get_onehot <- function(pred_val, truth_val) {
    as.numeric(grid$pred == pred_val & grid$truth == truth_val)
  }
  
  Z_mat <- t(mapply(get_onehot,
                    pred_val = as.character(dat$y_pred),
                    truth_val = as.character(dat$y_true)))
  
  p_hat <- colMeans(Z_mat)
  
  ## Naive iid multinomial covariance for one observation
  Sigma1_hat <- diag(p_hat) - tcrossprod(p_hat)
  
  g_fun <- function(p) metric_value(p, classes = classes, metric = metric)
  grad_hat <- grad(func = g_fun, x = p_hat)
  
  theta_hat <- g_fun(p_hat)
  V_hat_naive <- as.numeric(t(grad_hat) %*% Sigma1_hat %*% grad_hat)
  se_hat_naive <- sqrt(V_hat_naive / N)
  
  list(
    classes = classes,
    metric = metric,
    N = N,
    p_hat = p_hat,
    theta_hat = theta_hat,
    Sigma1_hat = Sigma1_hat,
    grad_hat = grad_hat,
    V_hat_naive = V_hat_naive,
    se_hat_naive = se_hat_naive
  )
}

## For paired models
estimate_paired_models_naive <- function(data_pair,
                                         classes = sort(unique(c(data_pair$y_true,
                                                                 data_pair$y_pred_ref,
                                                                 data_pair$y_pred_cand))),
                                         metric = c("micro_f1", "macro_f1")) {
  metric <- match.arg(metric)
  classes <- sort(classes)
  r <- length(classes)
  
  grid3 <- expand.grid(
    pred_cand = classes,
    pred_ref  = classes,
    truth     = classes,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  dat <- data_pair %>%
    mutate(
      y_true = factor(y_true, levels = classes),
      y_pred_ref = factor(y_pred_ref, levels = classes),
      y_pred_cand = factor(y_pred_cand, levels = classes)
    )
  
  N <- nrow(dat)
  
  get_onehot3 <- function(pc, pr, tr) {
    as.numeric(grid3$pred_cand == pc &
                 grid3$pred_ref  == pr &
                 grid3$truth     == tr)
  }
  
  W_mat <- t(mapply(get_onehot3,
                    pc = as.character(dat$y_pred_cand),
                    pr = as.character(dat$y_pred_ref),
                    tr = as.character(dat$y_true)))
  
  q_hat <- colMeans(W_mat)
  
  # Naive iid multinomial covariance for one paired observation
  Sigma1_hat <- diag(q_hat) - tcrossprod(q_hat)
  
  q_to_p_cand <- function(q) {
    p <- numeric(r^2)
    grid2 <- make_confusion_grid(classes)
    for (k in seq_len(nrow(grid2))) {
      a <- grid2$pred[k]
      b <- grid2$truth[k]
      idx <- which(grid3$pred_cand == a & grid3$truth == b)
      p[k] <- sum(q[idx])
    }
    p
  }
  
  q_to_p_ref <- function(q) {
    p <- numeric(r^2)
    grid2 <- make_confusion_grid(classes)
    for (k in seq_len(nrow(grid2))) {
      a <- grid2$pred[k]
      b <- grid2$truth[k]
      idx <- which(grid3$pred_ref == a & grid3$truth == b)
      p[k] <- sum(q[idx])
    }
    p
  }
  
  h_fun <- function(q) {
    p_c <- q_to_p_cand(q)
    p_r <- q_to_p_ref(q)
    metric_value(p_c, classes = classes, metric = metric) -
      metric_value(p_r, classes = classes, metric = metric)
  }
  
  d_hat <- h_fun(q_hat)
  grad_hat <- grad(func = h_fun, x = q_hat)
  
  Vd_hat_naive <- as.numeric(t(grad_hat) %*% Sigma1_hat %*% grad_hat)
  se_d_hat_naive <- sqrt(Vd_hat_naive / N)
  
  list(
    classes = classes,
    metric = metric,
    N = N,
    q_hat = q_hat,
    d_hat = d_hat,
    Sigma1_hat = Sigma1_hat,
    grad_hat = grad_hat,
    Vd_hat_naive = Vd_hat_naive,
    se_d_hat_naive = se_d_hat_naive
  )
}

## ---------------------------------------------------------
## 8. Power and sample size formulas
## ---------------------------------------------------------
superiority_power <- function(N_total, theta1, theta0, V, alpha = 0.05) {
  pnorm(sqrt(N_total) * (theta1 - theta0) / sqrt(V) - qnorm(1 - alpha))
}

superiority_sample_size <- function(theta1, theta0, V, alpha = 0.05, beta = 0.20) {
  ((qnorm(1 - alpha) + qnorm(1 - beta))^2 * V) / (theta1 - theta0)^2
}

noninferiority_power <- function(N_total, d1, Delta, Vd, alpha = 0.05) {
  pnorm(sqrt(N_total) * (d1 + Delta) / sqrt(Vd) - qnorm(1 - alpha))
}

noninferiority_sample_size <- function(d1, Delta, Vd, alpha = 0.05, beta = 0.20) {
  ((qnorm(1 - alpha) + qnorm(1 - beta))^2 * Vd) / (d1 + Delta)^2
}

## ---------------------------------------------------------
## 9. Inference procedures on Stage2_testing
## ---------------------------------------------------------
## For superiority (cluster robust)
superiority_inference <- function(theta_hat, V_hat, N, theta0, alpha = 0.05) {
  se_hat <- sqrt(V_hat / N)
  z_stat <- sqrt(N) * (theta_hat - theta0) / sqrt(V_hat)
  p_value <- 1 - pnorm(z_stat)
  
  ci_two_sided <- c(
    theta_hat - qnorm(1 - alpha / 2) * se_hat,
    theta_hat + qnorm(1 - alpha / 2) * se_hat
  )
  
  lower_one_sided <- theta_hat - qnorm(1 - alpha) * se_hat
  
  list(
    theta_hat = theta_hat,
    se_hat = se_hat,
    z_stat = z_stat,
    p_value = p_value,
    ci_two_sided = ci_two_sided,
    lower_one_sided = lower_one_sided
  )
}

## For inferiority (cluster robust)
noninferiority_inference <- function(d_hat, Vd_hat, N, Delta, alpha = 0.05) {
  se_hat <- sqrt(Vd_hat / N)
  z_stat <- sqrt(N) * (d_hat + Delta) / sqrt(Vd_hat)
  p_value <- 1 - pnorm(z_stat)
  
  ci_two_sided <- c(
    d_hat - qnorm(1 - alpha / 2) * se_hat,
    d_hat + qnorm(1 - alpha / 2) * se_hat
  )
  
  lower_one_sided <- d_hat - qnorm(1 - alpha) * se_hat
  
  list(
    d_hat = d_hat,
    se_hat = se_hat,
    z_stat = z_stat,
    p_value = p_value,
    ci_two_sided = ci_two_sided,
    lower_one_sided = lower_one_sided
  )
}

## Naive inference (iid)
superiority_inference_naive <- function(theta_hat, V_hat_naive, N, theta0, alpha = 0.05) {
  superiority_inference(theta_hat = theta_hat, V_hat = V_hat_naive, N = N, theta0 = theta0, alpha = alpha)
}

noninferiority_inference_naive <- function(d_hat, Vd_hat_naive, N, Delta, alpha = 0.05) {
  noninferiority_inference(d_hat = d_hat, Vd_hat = Vd_hat_naive, N = N, Delta = Delta, alpha = alpha)
}

## ---------------------------------------------------------
## 10. Stage 1: pilot design calculations
## ---------------------------------------------------------
alpha <- 0.05
beta  <- 0.10  

## Main metric for illustration
main_metric <- "macro_f1"

## Superiority
pilot_cand_Stage1 <- estimate_single_model(
  Stage1_cand_data,
  metric = main_metric
)

theta0 <- 0.755
theta1 <- pilot_cand_Stage1$theta_hat

N_required_sup <- superiority_sample_size(
  theta1 = theta1,
  theta0 = theta0,
  V = pilot_cand_Stage1$V_hat,
  alpha = alpha,
  beta = beta
)

n_required_sup <- ceiling(N_required_sup / pilot_cand_Stage1$mbar_hat); n_required_sup

cat("\n==============================\n")
cat("Stage 1: SUPERIORITY DESIGN\n")
cat("==============================\n")
cat("Metric:", main_metric, "\n")
cat("Pilot theta_hat =", round(theta1, 4), "\n")
cat("Pilot V_hat     =", round(pilot_cand_Stage1$V_hat, 6), "\n")
cat("theta0          =", theta0, "\n")
cat("Required total observations =", ceiling(N_required_sup), "\n")
cat("Required subjects           =", n_required_sup, "\n")

## Power curve for superiority
subject_grid <- seq(5, 35, by = 1)

power_df_sup <- data.frame(
  n_subjects = subject_grid,
  N_total = subject_grid * pilot_cand_Stage1$mbar_hat
) %>%
  mutate(
    power = superiority_power(
      N_total = N_total,
      theta1 = theta1,
      theta0 = theta0,
      V = pilot_cand_Stage1$V_hat,
      alpha = alpha
    )
  )

p_sup <- ggplot(power_df_sup, aes(x = n_subjects, y = power)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = (1-beta), linetype = 2) +
  labs(
    title = "Stage 1 pilot-based power curve for superiority testing",
    subtitle = paste("Metric:", main_metric, "| Proposed candidate RF (ntree = 10)"),
    x = "Number of subjects",
    y = "Expected power"
  ) +
  theme_minimal()

print(p_sup)

## Noninferiority: candidate vs reference
pilot_pair_Stage1 <- estimate_paired_models(
  Stage1_pair_data,
  metric = main_metric
)

Delta <- 0.036
d1 <- pilot_pair_Stage1$d_hat

N_required_ni <- noninferiority_sample_size(
  d1 = d1,
  Delta = Delta,
  Vd = pilot_pair_Stage1$Vd_hat,
  alpha = alpha,
  beta = beta
)

n_required_ni <- ceiling(N_required_ni / pilot_pair_Stage1$mbar_hat);n_required_ni

cat("\n=====================================\n")
cat("Stage 1: NONINFERIORITY DESIGN\n")
cat("=====================================\n")
cat("Metric:", main_metric, "\n")
cat("Pilot d_hat   =", round(d1, 4), "\n")
cat("Pilot Vd_hat  =", round(pilot_pair_Stage1$Vd_hat, 6), "\n")
cat("Delta         =", Delta, "\n")
cat("Required total observations =", ceiling(N_required_ni), "\n")
cat("Required subjects           =", n_required_ni, "\n")

## Power curve for noninferiority
power_df_ni <- data.frame(
  n_subjects = subject_grid,
  N_total = subject_grid * pilot_pair_Stage1$mbar_hat
) %>%
  mutate(
    power = noninferiority_power(
      N_total = N_total,
      d1 = d1,
      Delta = Delta,
      Vd = pilot_pair_Stage1$Vd_hat,
      alpha = alpha
    )
  )

p_ni <- ggplot(power_df_ni, aes(x = n_subjects, y = power)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = (1-beta), linetype = 2) +
  labs(
    title = "Stage 1 pilot-based power curve for noninferiority testing",
    subtitle = paste("Metric:", main_metric, "| Proposed RF (ntree = 10) vs Gold standard RF (ntree = 100)"),
    x = "Number of subjects",
    y = "Expected power"
  ) +
  theme_minimal()

print(p_ni)

## Combined plot
power_df_sup$type <- "Superiority"
power_df_ni$type  <- "Noninferiority"

power_df_all <- rbind(power_df_sup, power_df_ni)

ggplot(power_df_all, aes(x = n_subjects, y = power, color = type)) +
  geom_line(linewidth = 1) +
  
  ## Horizontal target power line
  geom_hline(yintercept = (1-beta), linetype = "dashed", color = "black") +
  
  ## Vertical lines for required sample sizes
  geom_vline(xintercept = n_required_ni, linetype = "dotted", color = "red") +
  geom_vline(xintercept = n_required_sup, linetype = "dotted", color = "blue") +
  
  ## Annotations
  annotate("text", x = n_required_ni, y = 0.2, label = paste0("n = ",n_required_ni,"\n(Noninferiority)"), 
           angle = 0, vjust = -0.5, size = 3.5, color = "red") +
  annotate("text", x = n_required_sup, y = 0.2, label = paste0("n = ",n_required_sup,"\n(Superiority)"), 
           angle = 0, vjust = -0.5, size = 3.5, color = "blue") +
  
  labs(
    title = "Power Curves for Superiority and Noninferiority Testing",
    subtitle = "Pilot-based sample size determination under clustered observations",
    x = "Number of subjects",
    y = "Expected power",
    color = "Test type"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

## ---------------------------------------------------------
## 11. Stage 2: final validation inference
## ---------------------------------------------------------
## Superiority
final_ref_Stage2 <- estimate_single_model(
  Stage2_ref_data,
  metric = main_metric
)

sup_out <- superiority_inference(
  theta_hat = final_ref_Stage2$theta_hat,
  V_hat = final_ref_Stage2$V_hat,
  N = final_ref_Stage2$N,
  theta0 = theta0,
  alpha = alpha
)

final_ref_Stage2_naive <- estimate_single_model_naive(
  Stage2_ref_data,
  metric = main_metric
)

sup_out_naive <- superiority_inference_naive(
  theta_hat = final_ref_Stage2_naive$theta_hat,
  V_hat_naive = final_ref_Stage2_naive$V_hat_naive,
  N = final_ref_Stage2_naive$N,
  theta0 = theta0,
  alpha = alpha
)

cat("\n=====================================\n")
cat("Stage 2: FINAL SUPERIORITY INFERENCE\n")
cat("=====================================\n")
cat("Metric:", main_metric, "\n")

cat("\n[Cluster-robust]\n")
cat("theta_hat =", round(sup_out$theta_hat, 4), "\n")
cat("SE        =", round(sup_out$se_hat, 4), "\n")
cat("Z         =", round(sup_out$z_stat, 4), "\n")
cat("p-value   =", signif(sup_out$p_value, 4), "\n")
cat("95% CI    = [", round(sup_out$ci_two_sided[1], 4), ", ",
    round(sup_out$ci_two_sided[2], 4), "]\n", sep = "")
cat("One-sided 95% lower bound = ", round(sup_out$lower_one_sided, 4), "\n", sep = "")

cat("\n[Naive iid]\n")
cat("theta_hat =", round(sup_out_naive$theta_hat, 4), "\n")
cat("SE        =", round(sup_out_naive$se_hat, 4), "\n")
cat("Z         =", round(sup_out_naive$z_stat, 4), "\n")
cat("p-value   =", signif(sup_out_naive$p_value, 4), "\n")
cat("95% CI    = [", round(sup_out_naive$ci_two_sided[1], 4), ", ",
    round(sup_out_naive$ci_two_sided[2], 4), "]\n", sep = "")
cat("One-sided 95% lower bound = ", round(sup_out_naive$lower_one_sided, 4), "\n", sep = "")

## Noninferiority: candidate vs reference
final_pair_Stage2 <- estimate_paired_models(
  Stage2_pair_data,
  metric = main_metric
)

ni_out <- noninferiority_inference(
  d_hat = final_pair_Stage2$d_hat,
  Vd_hat = final_pair_Stage2$Vd_hat,
  N = final_pair_Stage2$N,
  Delta = Delta,
  alpha = alpha
)

final_pair_Stage2_naive <- estimate_paired_models_naive(
  Stage2_pair_data,
  metric = main_metric
)

ni_out_naive <- noninferiority_inference_naive(
  d_hat = final_pair_Stage2_naive$d_hat,
  Vd_hat_naive = final_pair_Stage2_naive$Vd_hat_naive,
  N = final_pair_Stage2_naive$N,
  Delta = Delta,
  alpha = alpha
)

cat("\n=========================================\n")
cat("Stage 2: FINAL NONINFERIORITY INFERENCE\n")
cat("=========================================\n")
cat("Metric:", main_metric, "\n")

cat("\n[Cluster-robust]\n")
cat("d_hat (Proposed - Gold standard) =", round(ni_out$d_hat, 4), "\n")
cat("SE                               =", round(ni_out$se_hat, 4), "\n")
cat("Z                                =", round(ni_out$z_stat, 4), "\n")
cat("p-value                          =", signif(ni_out$p_value, 4), "\n")
cat("95% CI                           = [", round(ni_out$ci_two_sided[1], 4), ", ",
    round(ni_out$ci_two_sided[2], 4), "]\n", sep = "")
cat("One-sided 95% lower bound        = ", round(ni_out$lower_one_sided, 4), "\n", sep = "")

cat("\n[Naive iid]\n")
cat("d_hat (Proposed - Gold standard) =", round(ni_out_naive$d_hat, 4), "\n")
cat("SE                               =", round(ni_out_naive$se_hat, 4), "\n")
cat("Z                                =", round(ni_out_naive$z_stat, 4), "\n")
cat("p-value                          =", signif(ni_out_naive$p_value, 4), "\n")
cat("95% CI                           = [", round(ni_out_naive$ci_two_sided[1], 4), ", ",
    round(ni_out_naive$ci_two_sided[2], 4), "]\n", sep = "")
cat("One-sided 95% lower bound        = ", round(ni_out_naive$lower_one_sided, 4), "\n", sep = "")

## ---------------------------------------------------------
## 12. Optional: summarize key Stage sizes
## ---------------------------------------------------------
cat("\n=========================================\n")
cat("Stage SUMMARY\n")
cat("=========================================\n")
cat("Stage 1 training subjects:", length(unique(Stage1_train_subjects)), "\n")
cat("Stage 1 testing subjects :", length(unique(Stage1_test_subjects)), "\n")
cat("Stage 2 testing subjects :", length(unique(Stage2_test_subjects)), "\n")
cat("Stage 1 testing obs      :", nrow(Stage1_ref_data), "\n")
cat("Stage 2 testing obs      :", nrow(Stage2_ref_data), "\n")

summary_compare <- data.frame(
  Method = c("Cluster-robust", "Naive iid"),
  SE_superiority = c(sup_out$se_hat, sup_out_naive$se_hat),
  p_superiority = c(sup_out$p_value, sup_out_naive$p_value),
  CI_low_superiority = c(sup_out$ci_two_sided[1], sup_out_naive$ci_two_sided[1]),
  CI_high_superiority = c(sup_out$ci_two_sided[2], sup_out_naive$ci_two_sided[2]),
  SE_noninferiority = c(ni_out$se_hat, ni_out_naive$se_hat),
  p_noninferiority = c(ni_out$p_value, ni_out_naive$p_value),
  CI_low_noninferiority = c(ni_out$ci_two_sided[1], ni_out_naive$ci_two_sided[1]),
  CI_high_noninferiority = c(ni_out$ci_two_sided[2], ni_out_naive$ci_two_sided[2])
)

print(summary_compare)
