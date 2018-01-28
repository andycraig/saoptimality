#include "RcppArmadillo.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//' @param a x, y coordinates of first point.
//' @param b x, y coordinates of second point.
// [[Rcpp::export]]
double euclidean_distance(arma::rowvec a, arma::rowvec b) {
  return(sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)));
}

//' Wrap INLA::inla.matern.cov(). Only accepts a single distance (not vectorised at all).
//' @param x The distance between the two points.
// [[Rcpp::export]]
double matern_corr(double nu, double kappa, double x) {
  Rcpp::Environment package_env("package:INLA");
  Rcpp::Function rfunction = package_env["inla.matern.cov"];
  // R signature:
  // inla.matern.cov(nu, kappa, x, d = 1, corr = FALSE, norm.corr = FALSE, theta, epsilon = 1e-08)
  return(as<double>(rfunction(nu, kappa, x, 2, true)));
}

//' @param D N x 2 matrix, with each row being a location.
//' @param X N x N matrix to fill with pairwise distances.
// [[Rcpp::export]]
arma::mat get_dist_matrix(const arma::mat& D) {
  arma::mat X = arma::mat(D.n_rows, D.n_rows, arma::fill::zeros);
  for (unsigned int i_elem = 0; i_elem < D.n_rows - 1; ++i_elem) {
    for (unsigned int j_elem = i_elem + 1; j_elem < D.n_rows; ++j_elem) {
      X(j_elem, i_elem) = sqrt(sum(pow((D.row(i_elem) - D.row(j_elem)),2)));
      X(i_elem, j_elem) = X(j_elem, i_elem);
    }
  }
  return (X);
}

//' Do a fast rank-one update of an inverted matrix,
//' using the Sherman–Morrison formula, such that:
//' A^-1 -> (A + u v)^-1
//' @param A_inv n x n matrix, already inverted.
//' @param u n x 1 matrix.
//' @param v n x 1 matrix.
// [[Rcpp::export]]
void rank_one_update(arma::mat& A_inv, arma::vec& u, arma::vec& v) {
  double denom = as_scalar(1 + v.t() * A_inv * u); 
  A_inv = A_inv - (A_inv * u * v.t() * A_inv) / denom; 
}

//' Update a row and column of a matrix.
//' A -> A with row/col j replaced
//' @param want The new column (and row).
//' @param j The row/col of \code{A} to be replaced.
// [[Rcpp::export]]
void row_col_update_of_matrix(arma::mat& A, arma::vec want, int j) {
  // Update A.
  A.col(j) = want;
  A.row(j) = want.t();
}

//' Efficiently update a matrix's inverse when one of the matrix's 
//' rows and columns is updated.
//' A^-1 -> (A with row/col j replaced)^-1
//' @param want The new column (and row).
//' @param j The row/col of \code{A} to be replaced.
// [[Rcpp::export]]
void row_col_update_of_inverse(const arma::mat& A, arma::mat& A_inv, arma::vec want, int j) {
  // Update A_inv.
  // Do via two rank-one updates.
  arma::vec u = arma::zeros<arma::vec>(A_inv.n_rows);
  u[j] = 1;
  arma::vec v = want - A.col(j);
  rank_one_update(A_inv, u, v);
  // Update row j.
  arma::vec u2 = want - A.col(j);
  u2[j] = 0; // Zero out jth element to avoid subtracting (j,j)th element of A twice.
  // 'v2' is same as u, so don't need to create.
  rank_one_update(A_inv, u2, u);
}

//' Efficiently update both the matrix and its inverse  when one of its 
//' rows and columns is updated.
//' A -> A with row/col j replaced
//' A^-1 -> (A with row/col j replaced)^-1
//' @param want The desired vector to be put in row/col \code{j}.
//' @param j Row/col of A to update, zero-indexed.
// [[Rcpp::export]]
void row_col_update_of_matrix_and_inverse(arma::mat& A, arma::mat& A_inv, arma::vec want, int j) {
  row_col_update_of_inverse(A, A_inv, want, j);
  row_col_update_of_matrix(A, want, j);
}

//' Convert selection weights to selection probabilities,
//' 'shaped' by \code{temperature}. such that if \code{temperature} is
//' one, probabilities are simply the weights normalised, and if 
//' \code{temperature} was zero all non-zero weights would have equal probability
//' (although temperature should never reach zero in simulated annealing
//' and so this will actually raise an error.
//' @param w NumericVector. The weights.
//' @param  temperature Double in range (0, 1].
//' @example 
//' get_annealed_prob(c(0.5, 0, 0.2), 0.0)
//' get_annealed_prob(c(0.5, 0, 0.2), 0.5)
//' get_annealed_prob(c(0.5, 0, 0.2), 1.0)
NumericVector get_annealed_prob(NumericVector w, double temperature) {
  // If temperature is 0, all non-zero elements should have equal probability.
  // But temperature should never reach 0.
  //TODO Avoid double equality.
  if (temperature <= 0) {
    ::Rf_error("temperature <= 0 not allowed");
  }
  if (temperature > 1) {
    ::Rf_error("temperature > 1 not allowed");
  }
  // Construct prob = (w^temperature / sum(w^temperature)
  int n_w = w.size();
  double denominator = sum(pow(w, temperature));
  NumericVector prob(n_w);
  for (int i_w = 0; i_w < n_w; ++i_w) {
    prob[i_w] = pow(w(i_w), temperature) / denominator;
  }
  return(prob);
}

// Class that records current state of simulated annealing particle.
// Has methods for modifying itself to a new proposed state, and returning
// to its previous state.
// Energy function is an alphabet optimality criterion involving the 
// inverse of a covariance matrix. This is cached to improve evaluation speed.
class State {
private:
  const int family; // Gaussian, logistic etc.
  const double resolution; // The minimum resolution of the locations.
  const double nu; // Matern covariance parameter.
  const double kappa; // Matern covariance parameter.
  const arma::vec betas;
  //TODO Make X, D const without throwing errors when .rows() is called on them.
  arma::mat X; // N x p matrix of the covariates of ALL the available candidates.
  arma::mat D; // N x 2 matrix of the x, y locations of ALL the available candidates.
  const arma::uvec grps; // Same length as candidates. Each element gives the group of the corresponding element of s.
  const std::vector<arma::mat> weights;
  const arma::uvec indices_within_weights; // For finding correct values in weights.
  const bool exclusive;
  const double ar1_rho;
  const int t; // Number of time points.
  double s2rf; // Variance of random field.
  arma::uvec Ds_non_parameters; // Mask for betas not of interest, for D_s optimality.
  bool is_Ds; // true to calculate Ds optimality, false for D optimality.
  arma::mat C_temporal, C_temporal_inv, C_spatial, C_spatial_inv, W; // For caching parts of optimality calculation.
  bool can_reject; // Flag for whether proposal has been made.
  arma::uvec s; // The elements currently selected to be in the state. Unordered.
  IntegerVector indexes_in_s;
  arma::mat s_X; // The subset of rows of X corresponding to just those elements in s.
  arma::mat s_D; // The subset of rows of D corresponding to just those elements in s.
  unsigned int index_in_s_to_switch; // Tracks the index of the element in s for which changes are currently being proposed/rejected.
  unsigned int old_element; // The old value of the element of s.
  double e, e_old, e_old_old;
  // When s_X (design matrix of current state) is updated, multiple rows need to change.
  // This is a utility function to do this.
  void update_s_X(int index_in_s, int element) {
    // Change rows in s_X corresponding to index_in_s_to_switch 
    // to rows in X corresponding to element.
    // Need to update t rows of s_X.
    int s_X_start = index_in_s * t;
    int s_X_end = (index_in_s + 1) * t - 1;
    int X_start = element * t;
    int X_end = (element + 1) * t - 1;
    if (s_X_start == s_X_end) {
      // Only one row - can't use .rows().
      s_X.row(s_X_start) = X.row(X_start);
    } else {
      s_X.rows(s_X_start, s_X_end) = X.rows(X_start, X_end);
    }
  }
  // Updates components of D-optimality affected by change of state: 
  // - Covariance betwen locations.
  // - Design matrix.
  // - Optimality score.
  void update_d_optimality(int element) {
    s[index_in_s_to_switch] = element;
    // Update s_D (locations).
    s_D.row(index_in_s_to_switch) = D.row(s[index_in_s_to_switch]);
    arma::vec new_C_row = locations2corrs();
    // Since only one element is updated at a time, we need only update one 
    // row and column of each of X and C (and hence C_inv). 
    // Update X.
    update_s_X(index_in_s_to_switch, element);
    // Update C, C^-1. 
    row_col_update_of_matrix_and_inverse(C_spatial, C_spatial_inv, new_C_row, index_in_s_to_switch);
    
    // Update W. 
    // If family == 0 (Gaussian), W is always identity and don't need to do anything.
    if (family == 1) { // Logistic/binomial
      // Need to update t elements.
      //TODO Update t rows of W.
      int s_X_start = index_in_s_to_switch * t;
      int s_X_end = (index_in_s_to_switch + 1) * t - 1;
      arma::vec p_y_j = 1 / (1 + exp(-1 * s_X.rows(s_X_start, s_X_end) * betas));
      W(arma::span(s_X_start, s_X_end), arma::span(s_X_start, s_X_end)) = arma::diagmat(p_y_j % (1 - p_y_j));
    }
    calculate_d_optimality();
  }
  // Calculate D-optimality in separate function so it can be called 
  // on initialisation and on update.
  void calculate_d_optimality() {
    // Re-compute optimality criterion.
    // We seek to MINIMISE this criterion e. Small means a larger information matrix determinant.
    // Rcout << "Calculating D-optimality..." << std::endl;
    // Rcout << "s_X" << std::endl<< s_X << std::endl;
    // Rcout << "C_temporal_inv" << std::endl<< C_temporal_inv << std::endl;
    // Rcout << "C_spatial_inv" << std::endl<< C_spatial_inv << std::endl;
    // Rcout << "C" << std::endl<< kron(C_temporal_inv, C_spatial_inv) << std::endl;
    // Rcout << "W " << std::endl << W << std::endl;
    //Rcout << "Information matrix before taking det " << std::endl << s_X.t() * C_inv * W * s_X << std::endl;
    // Using identity that inv(s2rf * kron(A, B)) = 1 / s2rf * kron(inv(a), inv(B)).
    arma::mat numerator_matrix = 1 / s2rf * s_X.t() * kron(C_temporal_inv, C_spatial_inv) * W * s_X;
    if (is_Ds) {
      // D_s optimality.
      e = -1 * log(det(numerator_matrix) / det(numerator_matrix(Ds_non_parameters, Ds_non_parameters)));
    } else {
      // D optimality.
      e = -1 * log(det(numerator_matrix));
    }
  }
  arma::vec locations2corrs() {
    // Use index_in_s_to_switch to look up the element of s that has changed, 
    // and calculate the covariance between this and all other locations in s.
    arma::rowvec loc_of_index_in_s_to_switch = D.row(s[index_in_s_to_switch]);
    arma::vec covs(s_D.n_rows);
    for (unsigned int i_loc = 0; i_loc < covs.size(); ++i_loc) {
      if (i_loc == index_in_s_to_switch) {
        covs[i_loc] = 1; // Never actually used.
      } else {
        // If dist i/s 0, which can happen if exclusive is false, set it to a small number because the locations wouldn't be the same in practice.
        double eucl_dist = euclidean_distance(loc_of_index_in_s_to_switch, s_D.row(i_loc));
        double dist = std::max(resolution, eucl_dist);
        covs[i_loc] = matern_corr(nu, kappa, dist); 
      }
    }
    return (covs);
  }
public:
  // Constructor.
  // @param weights Each element of \code{weights} must have the same length as \code{s_init}.
  // In each element of \code{weights}, only those elements corresponding to a single group should be non-zero.
  // (This structure is for convenience of indexing.)
  State(arma::mat X_, arma::mat D_, bool exclusive_, arma::uvec grps_, arma::uvec s_,
        std::vector<arma::mat> weights_, arma::uvec indices_within_weights_, 
        double nu_, double kappa_, double resolution_, arma::vec betas_, int family_, 
        arma::uvec Ds_parameters, double ar1_rho_, int t_, double s2rf_) : 
  grps(grps_), weights(weights_), indices_within_weights(indices_within_weights_),
  exclusive(exclusive_), nu(nu_), kappa(kappa_), resolution(resolution_), 
  betas(betas_), family(family_), ar1_rho(ar1_rho_), t(t_), s2rf(s2rf_) {

    X = X_;
    D = D_;
    s = s_;
    can_reject = false;
    is_Ds = Ds_parameters.size() > 0; 
    if (is_Ds) {
      Ds_non_parameters = arma::uvec(betas.size(), arma::fill::ones);
      Ds_non_parameters(Ds_parameters) = arma::uvec(Ds_parameters.size(), arma::fill::zeros);
    }
    // Set indexes_in_s to 0:(length(s) - 1). 
    indexes_in_s = arma::linspace<arma::vec>(0, s.size() - 1, s.size());
    // Initialise subsets of X, D for the initial s.
    s_X = arma::mat(s.size() * t, betas.size());
    for (unsigned int i_s = 0; i_s < s.size(); ++i_s) {
      update_s_X(i_s, s(i_s));
    }
    s_D = D.rows(s);
    // Calculate correlation matrix for initial s.
    // Calculate spatial component.
    C_spatial = arma::mat(s.size(), s.size()); // Need to retain, for updating when units change.
    for (unsigned int i_row = 0; i_row < s.size(); ++i_row) {
      C_spatial(i_row, i_row) = 1; // The diagonal.
      for (unsigned int i_col = 0; i_col < i_row; ++i_col) { // Symmetric matrix, so just calculate lower triangle.
        C_spatial(i_row, i_col) = matern_corr(nu, kappa, std::max(resolution, euclidean_distance(s_D.row(i_row), s_D.row(i_col))));
        C_spatial(i_col, i_row) = C_spatial(i_row, i_col); // Symmetric matrix, so update the upper triangle.
      }
    }
    //TODO Deal with case in which C is singular, which can probably happen if all candidates are in same place.
    C_spatial_inv = inv(C_spatial);
    // Calculate temporal component.
    // Will never update as same for all units, but retain for output.
    if (t == 1) {
      C_temporal = arma::mat(1, 1, arma::fill::ones); 
    } else {
      C_temporal = arma::mat(t, t);
      for (unsigned int i_row = 0; i_row < C_temporal.size(); ++i_row) {
        C_temporal(i_row, i_row) = 1; // Diagonal.
        for (unsigned int i_col = 0; i_col < i_row; ++i_col) {
          C_temporal(i_row, i_col) = pow(ar1_rho, abs(i_row - i_col));
          C_temporal(i_col, i_row) = C_temporal(i_row, i_col);
        }
      }
    }
    C_temporal_inv = inv(C_temporal);
    
    // Calculate weight matrix W. 
    if (family == 0) { // Gaussian
      // Identity.
      W.eye(s_X.n_rows, s_X.n_rows);
    } else {
      if (family == 1) { // Logistic/binomial
        // It is a diagonal matrix, with the jth element being:
        // p(x_j; beta) (1 - p(x_j; beta),
        // where p(y) = 1 / (1 + exp(-y))
        arma::vec p_y = 1 / (1 + exp(-1 * s_X * betas));
        arma::vec w = p_y % (1 - p_y);
        W = arma::diagmat(w);
      }
    }
    
    // Compute optimality criterion.
    calculate_d_optimality(); // 
    e_old = e;
    e_old_old = e_old;
  }
  // Proposes a new state by switching one element.
  void propose(double temperature) {
    can_reject = true;
    index_in_s_to_switch = sample(indexes_in_s, 1)[0]; // The index of the item in s to switch.
    // Record the element we'll replace.
    old_element = s[index_in_s_to_switch]; // The value of the item in s to switch.
    // Want to replace with an element from the same group, so check that. 
    // Weight probabilities of indexes to select by 'distance' from s_to_switch.
    unsigned int grp_of_old_element = grps[old_element];
    arma::rowvec weights_available = weights[grp_of_old_element].row(indices_within_weights[old_element]);
    if (exclusive) {
      // Zero probability of all candidates in current state.
      for (unsigned int i_s = 0; i_s < s.size(); ++i_s) {
        if (grps[s[i_s]] == grp_of_old_element) {
          weights_available[indices_within_weights[s[i_s]]] = 0; 
        }
      }
    } else {
      weights_available[old_element] = 0; // Zero probability of selecting same candidate.
    }

    // Update s.
    //TODO Avoid cast to NumericVector.
    arma::uvec indices_of_old_grp = find(grps == grp_of_old_element);
    int new_candidate = sample(IntegerVector(indices_of_old_grp.begin(), indices_of_old_grp.end()), 
                                1, 
                                get_annealed_prob(NumericVector(weights_available.begin(), 
                                                                       weights_available.end()),
                                                                       temperature))[0];
    // Update e.
    e_old_old = e_old;
    e_old = e;
    update_d_optimality(new_candidate); // Updates e.
  }
  // Set s back to what it was before last call to propose().
  void reject () {
    if (can_reject) {
      can_reject = false;
      update_d_optimality(old_element);
      e = e_old;
      e_old = e_old_old;
    } else {
      ::Rf_error("Trying to reject without having made a proposal. Indicates bad program logic."); 
    }
  }
  // Get current s.
  IntegerVector get_s_clone() {return IntegerVector(s.begin(), s.end());}
  // Get current D-optimality components.
  arma::mat get_C_spatial() {return C_spatial;}
  arma::mat get_C_temporal() {return C_temporal;}
  arma::mat get_X() {return s_X;}
  arma::mat get_W() {return W;}
  // Get energy of current s.
  double evaluate() {return(e);}
  // Get energy of old s.
  double evaluate_previous() {return(e_old);}
};

//' Function for (possibly) getting new state.
//' @param s List, with each element being the state for that group.
//' @param candidates List, with each element being the candidates for that group.
//' @param weights List, with each element being the transition weight matrix for that group. (This will be adjusted to make probabilities.)
//' @param e Function to compute the energy.
//' @param e_s Numeric. Result of \code{e(s)}. If \code{NULL}, this will be computed, but it can be passed in to improve efficiency.
//' @param exclusive Boolean. If TRUE, the returned state can contain the same candidate multiple times.
//' @example 
//' s = list(a = c(1, 2), b = c(1, 2))
//' candidates = list(a = 1:4, b = 1:4)
//' weights = list(a = matrix(1, 4, 4) - diag(4),  b = matrix(1, 4, 4) - diag(4))
//' get_next_state(s, candidates, weights, e = function(x) sum(unlist(x)), temperature = 0.5)
double get_next_state(State& s, double temperature) {
  // Propose new candidate.
  s.propose(temperature);
  // Evaluate energies.
  double e_old = s.evaluate_previous();
  double e_proposed = s.evaluate();
  // Choose whether or not to accept.
  double acceptance_prob = (exp(-(e_proposed - e_old) / temperature));
  Rcout << "Old e: " << e_old << " Proposed e: " << e_proposed 
    << " Acceptance prob: " << acceptance_prob 
    << " Temperature: " << temperature << std::endl;
  if ((e_proposed < e_old) || (runif(1)[0] < acceptance_prob)) {
    // Accept proposed state. Just return the new energy.
    Rcout << "  Accepted proposed state." << std::endl;
    return(e_proposed);
  } else {
    // Don't accept proposed state. Switch state back to what it was.
    Rcout << "  Rejected proposed state." << std::endl;
    s.reject();
    // Return old energy.
    return(e_old);
  }
}

//' Choose cells to maximise an optimal experimental design objective,
//' which can involve spatio-temporal covariance. Each location may be specified as
//' belonging to a specific group, and the numbers of locations from each group 
//' may be specified.
//' 
//' Uses simulated annealing, and does the computationally heavy evaluation
//' of the energy function in an efficient manner.
//'
//' @param s The initial state, being some number of integers in the range [0, N-1]. May contain the same number multiple times if \code{exclusive} is \code{false}.
//' @param n_steps The number of simulated annealing steps to take.
//' @param nu; // Matern covariance parameter.
//' @param kappa; // Matern covariance parameter.
//' @param X (N x t) x p matrix of the covariates of ALL the available candidates. Rows must be 
//' organised by unit first then by time, e.g., 
//' id_0 time_0; id_0 time_1; ...; id_0 time_t; id_1; time_0; ...
//' @param D N x 2 matrix of the x, y locations of ALL the available candidates.
//' @param grps Same length as candidates. Each element gives the group of the corresponding element of s.
//' @param weights List, with each element being the transition weights for that group.
//' @param exclusive If \code{false}, the same candidate can be in the state multiple times.
//' @param betas p-length vector. Prior point estimates of regression coefficients, used to compute the weight matrix.
//' @param family 0 for Gaussian link, 1 for logistic link.
//' @param Ds_parameters If this contains any elements, the optimality criterion becomes  
//' D_s. For example: If Ds_parameters is [0, 2], then the 1st and 3rd (note the indexing) elements of 
//' beta are of interest. If Ds_parameters is empty, the optimality criterion is D.
//' @param ar1_rho The temporal autocorrelation.
//' @param t The number of time points.
//' @param s2rf The variance of the random field (which, when multiplied by the correlation of the 
//' random field, produces the covariance of the random field).
// [[Rcpp::export]]
List choose_cells_cpp(arma::mat X, arma::mat D, bool exclusive, arma::uvec grps,
                      arma::uvec s, double nu, double kappa, double resolution, 
                      arma::vec betas, int n_steps, int family, arma::uvec Ds_parameters,
                      double ar1_rho, int t, double s2rf) {
  if (exclusive) {
    arma::uvec s_unique = unique(s);
    if (s.size() != s_unique.size()) {
      stop("exclusive is true but there are duplicate values in initial state s.");
    }
  }
  arma::uvec grp_names = unique(grps);
  int n_grps = grp_names.size();
  
  // Create weights vector, each element of which is the distance matrix for a group.
  std::vector<arma::mat> weights(n_grps);
  // Create a uvec that can be used to look up an element's index within the weight matrix..
  arma::uvec indices_within_weights(grps.size(), arma::fill::zeros);
  for (int i_grp = 0; i_grp < n_grps; ++i_grp) {
    arma::uvec indices_for_this_grp = arma::find(grps == i_grp);
    weights[i_grp] = get_dist_matrix(D.rows(indices_for_this_grp));
    indices_within_weights.elem(indices_for_this_grp) = arma::linspace<arma::uvec>(0, indices_for_this_grp.size() - 1, indices_for_this_grp.size());
  }
  State state = State(X, D, exclusive, grps, s, weights, indices_within_weights, 
                      nu, kappa, resolution, betas, family, Ds_parameters, ar1_rho, 
                      t, s2rf);
  // Initialise best to current.
  IntegerVector s_best = state.get_s_clone();
  double e_initial = state.evaluate();
  double e_best = e_initial;
  
  // Simulated annealing loop.
  double e; // The energy.
  for (int i_step = 0; i_step < n_steps; ++i_step) {
    Rcout << "Step " << i_step + 1 << "/" << n_steps << "..." << std::endl;
    e = get_next_state(state, pow(0.99, i_step));
    // Retain state if it's the best we've seen.
    if (e < e_best) {
      s_best = state.get_s_clone();
      e_best = e;
    }
  }
  Rcout << "Done. Final e: " << e_best << " Initial e: " << e_initial << std::endl;
  
  // Return best state seen.
  return (Rcpp::List::create(Rcpp::Named("s") = s_best,
                             Rcpp::Named("e") = e_best,
                             Rcpp::Named("C_temporal") = state.get_C_temporal(),
                             Rcpp::Named("C_spatial") = state.get_C_spatial(),
                             Rcpp::Named("s2rf") = s2rf,
                             Rcpp::Named("X") = state.get_X(),
                             Rcpp::Named("W") = state.get_W()));
}
