#' Choose cells to maximise an optimal experimental design objective,
#' which can involve spatio-temporal covariance. Each location may be specified as
#' belonging to a specific group, and the numbers of locations from each group 
#' may be specified.
#' 
#' Uses simulated annealing, and does the computationally heavy evaluation
#' of the energy function in an efficient manner.
#' @param D N x 2 matrix with first column being x coordinates and second column y coordinates.
#' @param X (N x \code{t}) x p matrix of the covariates of ALL the available candidates. Rows must be 
#' organised by unit first then by time, e.g., 
#' id_0 time_0; id_0 time_1; ...; id_0 time_t; id_1; time_0; ... The chosen design 
#' matrix will be a subset of the rows of \code{X}.
#' @param numbers List with names being the values of \code{df}'s column \code{g}, and 
#' values being the number of items from that group to select from \code{df}.
#' @param n_steps Integer. The number of simulated annealing steps to take.
#' @param nu Numeric. The Matern smoothness parameter.
#' @param kappa Numeric. sqrt(8 * nu) / range, where range is usually the distance where the covariance is about 0.1.
#' @param resolution Numeric. The assumed resolution of \code{D}. In the case where
#' the state contains two units at the same location (e.g., if `\code{exclusive} is \code{FALSE})
#' a fraction of this will be added to one of them so that they aren't at exactly the same place.
#' @param betas Numeric vector of length p. Estimates of the regression coefficients. Used
#' to computer the weight matrix if \code{family} is not "gaussian".
#' @param s2rf The variance of the random field (which, when multiplied by the correlation of the 
#' random field, produces the covariance of the random field).
#' @param groups Integer vector. Either length N, with the ith element being the group of the ith unit, 
#' or \code{NULL}, which is equivalent to all elements belonging to the same group.
#' @param exclusive If \code{FALSE}, the same candidate can be in the state multiple times.
#' @param s_initial Integer vector or \code{NULL}. If an integer vector, the (1-indexed) indexes of 
#' \code{X}, \code{D} and \code{groups}. If \code{NULL}, initial state is chosen randomly. 
#' @param Ds_parameters Numeric vector. If this contains any elements, the optimality criterion becomes  
#' D_s. For example: If Ds_parameters is c(1, 3), then the 1st and 3rd (note the indexing) elements of 
#' beta are of interest. If Ds_parameters is empty or \code{NULL}, the optimality criterion is D.
#' One of \code{state_size}, \code{s_initial} must be specified.
#' @param ar1_rho Numeric. Temporal autocorrelation.
#' @param t Integer. Number of time points. (Could be inferred from sizes of \code{D} and \code{X}, 
#' but requiring to be specified as a sanity check.)
#' @value Integer vector (1-indexed) being the indexes of the \code{D}, \code{X}, \code{groups} 
#' that optimise the optimality criteria.
#' @example 
#' # Very simple 2 x 2 example
#' numbers = list(a = 1, b = 1)
#' x_max = 1
#' D = expand.grid(x = 0:x_max, y = 0:x_max)
#' X = matrix(runif((x_max + 1)^2), nrow = x_max + 1) # All units have same covariates (other than location).
#' groups = rep(c("a", "b"), each = nrow(D) / 2)
#' betas = c(1, 1); resolution = 0.5; s_initial = NULL; exclusive = TRUE
#' nu = 1
#' range = 1 # Distance where covariance becomes about 0.1
#' kappa =  sqrt(8 * nu) / range # INLA definition.
#' resolution = 0.5
#' family = "gaussian"
#' n_steps = 100
#' # Visualise available candidates.
#' library(ggplot2)
#' D %>% ggplot(aes(x = x , y = y, color = groups)) + geom_point()
#' # Optimise.
#' result = choose_cells(D, X, numbers, n_steps, nu, kappa, resolution, betas, s2rf, groups, exclusive, family = family, t = t, ar1_rho = ar1_rho) 
#' # Display initial.
#' D[result$s_initial,] %>% ggplot(aes(x = x, y = y, colour = groups[result$s_initial])) + geom_point(size = 10) + xlim(min(D$x), max(D$x)) + ylim(min(D$y), max(D$y))
#' # Display results. 
#' D[result$s,] %>% ggplot(aes(x = x, y = y, colour = groups[result$s])) + geom_point(size = 10) + xlim(min(D$x), max(D$x)) + ylim(min(D$y), max(D$y))
#' # Example with x coordinate being a covariate, very low spatial correlation, and Gaussian link.
#' # Points should be evenly distributed at the boundaries.
#' t = 1
#' numbers = list(a = 3)
#' nu = 1
#' range = 0.01 # Distance where covariance becomes about 0.1
#' kappa =  sqrt(8 * nu) / range # INLA definition.
#' D = expand.grid(x = seq(-1, 1, by = 0.2), y = seq(-1, 1, by = 0.2))
#' X = cbind(rep(1, nrow(D)), D$x) # Intercept and x coordinate as covariate.
#' betas = c(1, 1)
#' s2rf = 1
#' resolution = 0.1
#' exclusive = FALSE
#' family = "gaussian"
#' groups = rep("a", nrow(D))
#' ar1_rho = 0
#' n_steps = 1000
#' 
#' # Example with no covariates, high spatial correlation, and Gaussian link.
#' # Points should be far from each other.
#' t = 1
#' numbers = list(a = 4)
#' nu = 1
#' range = 2 # Distance where covariance becomes about 0.1
#' kappa =  sqrt(8 * nu) / range # INLA definition.
#' D = expand.grid(x = seq(-1, 1, by = 0.2), y = seq(-1, 1, by = 0.2))
#' X = matrix(1, nrow = nrow(D)) # Intercept and x coordinate as covariate.
#' betas = c(1) # Doesn't matter for Gaussian.
#' resolution = 0.1
#' exclusive = FALSE
#' family = "gaussian"
#' groups = rep("a", nrow(D))
#' ar1_rho = 0
#' 
#' # Example with x coordinate being a covariate, very low spatial correlation, and binomial link.
#' numbers = list(a = 3)
#' range = 0.01 # Distance where covariance becomes about 0.1
#' kappa =  sqrt(8 * nu) / range # INLA definition.
#' D = expand.grid(x = seq(-1, 1, by = 0.2), y = seq(-1, 1, by = 0.2))
#' X = cbind(rep(1, nrow(D)), D$x) # Intercept and x coordinate as covariate.
#' betas = c(1, 2) # Matters.
#' resolution = 0.1
#' exclusive = FALSE
#' family = "binomial"
#' groups = rep("a", nrow(D))
#' ar1_rho = 0
#' @export
choose_cells = function(D, X, numbers, n_steps, nu, kappa, resolution, betas, s2rf,
                        groups = NULL, exclusive = FALSE, s_initial = NULL, 
                        family = c("gaussian", "binomial"), Ds_parameters = NULL,
                        ar1_rho = NULL, t = NULL) {

  library(magrittr)
  library(INLA) # This is REQUIRED in order for the inla.matern.cov function to work.
  #TODO Fix library(INLA) being required.
  # Check inputs.
  family <- match.arg(family)
  if ((nrow(D) * t) != nrow(X)) {f
    stop(paste0("t (", t, ") times number of rows of D (", nrow(D), ") must be same as number of rows of X (", nrow(X), ").")) 
  }
  if (nrow(D) != length(groups)) {
    stop(paste0("Number of rows of D (", nrow(D), ") and length of groups (", length(groups), ") must be same.")) 
  }
  
  # Setup.
  family_int = which(family == c("gaussian", "binomial")) - 1 # -1 for 0-indexing.
  if (is.null(groups)) {
    groups = rep(0, nrow(X)) 
    group_names = 0
    groups_int = 0
  } else {
    # Convert group names to integers.
    group_names = unique(groups)
    if (!setequal(group_names, names(numbers))) {
      stop("Unique values in groups and names(numbers) must be the same.")
    }
    groups_lookup = 0:(length(group_names) - 1) %>% `names<-`(group_names)
    groups_int = groups_lookup[groups] %>% `names<-`(NULL)
  }
  
  if (is.null(s_initial)) {
    # Allocate intial state randomly.
    candidates = 1:length(groups_int) # 1-indexed. Will be converted to 0-indexed at cpp function call.
    for (grp in group_names) {
      grp_candidates = candidates[groups == grp]
      # sample behaves differently if first argument has length 1, so handle that case separately.
      if (length(grp_candidates) == 1) {
        s_initial = c(s_initial, rep(grp_candidates, numbers[[grp]]))
      } else { 
        s_initial = c(s_initial, sample(grp_candidates, numbers[[grp]], replace = !exclusive))
      }
    }
  }  
  
  message("Choosing cells...")
  # Call C++ function to do the work.
  # Its signature:
  # List choose_cells_cpp(arma::mat X, arma::mat D, bool exclusive, arma::uvec grps,
  # arma::uvec s, double nu, double kappa, double resolution, 
  # arma::vec betas, int n_steps)
  result = choose_cells_cpp(X, as.matrix(D), exclusive, groups_int, 
                            s_initial - 1, # To 0-indexed. 
                            nu, kappa, resolution, betas, n_steps, family_int, 
                            Ds_parameters - 1, # To 0-indexed.
                            ar1_rho, t, s2rf
  )
  # Values of s from choose_cells_cpp are 0-indexed. Change them to be 1-indexed.
  result$s = result$s + 1
  # Add s_initial to results.
  result$s_initial = s_initial + 1
  print("Finished choosing cell.")
  result
}

