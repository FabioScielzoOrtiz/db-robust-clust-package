#####################################################################################################################

import numpy as np
from typing import Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import train_test_split, KFold

from robust_mixed_dist.quantitative import (
    euclidean_dist, 
    euclidean_dist_matrix, 
    minkowski_dist,
    minkowski_dist_matrix, 
    canberra_dist,
    canberra_dist_matrix,
    mahalanobis_dist,
    mahalanobis_dist_matrix,
    robust_mahalanobis_dist,
    robust_mahalanobis_dist_matrix,
    S_robust
)
from robust_mixed_dist.binary import (
    sokal_dist, 
    sokal_dist_matrix,
    jaccard_dist,
    jaccard_dist_matrix
)
from robust_mixed_dist.multiclass import (
    hamming_dist, 
    hamming_dist_matrix
)
from robust_mixed_dist.mixed import (
    simple_gower_dist_matrix,
    simple_gower_dist,
    generalized_gower_dist_matrix,
    generalized_gower_dist,
    related_metric_scaling_dist_matrix_faster,
    geometric_variability
)

#####################################################################################################################
# 1. GLOBAL CONSTANTS
#####################################################################################################################

MIXED_METRICS = {'ggower', 'relms'}  # Set instead of List for O(1) lookups

DIST_MATRIX_FUNCS = {
    'euclidean': euclidean_dist_matrix, 
    'minkowski': minkowski_dist_matrix,
    'canberra': canberra_dist_matrix,
    'mahalanobis': mahalanobis_dist_matrix,
    'robust_mahalanobis': robust_mahalanobis_dist_matrix,
    'sokal': sokal_dist_matrix,
    'jaccard': jaccard_dist_matrix,
    'hamming': hamming_dist_matrix,
    'simple_gower': simple_gower_dist_matrix,
    'ggower': generalized_gower_dist_matrix, 
    'relms': related_metric_scaling_dist_matrix_faster
}

DIST_FUNCS = {
    'euclidean': euclidean_dist, 
    'minkowski': minkowski_dist,
    'canberra': canberra_dist,
    'mahalanobis': mahalanobis_dist,
    'robust_mahalanobis': robust_mahalanobis_dist,
    'sokal': sokal_dist,
    'jaccard': jaccard_dist,
    'hamming': hamming_dist,
    'simple_gower': simple_gower_dist,
    'ggower': generalized_gower_dist,
    # relms uses the same point-to-point base as ggower in this context
}

#####################################################################################################################
# 2. AUXILIARY FUNCTIONS
#####################################################################################################################

def extract_sample(
    X: np.ndarray, 
    y: Optional[np.ndarray] = None, 
    frac_sample_size: float = 0.2, 
    random_state: int = 42, 
    stratify: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts a stratified sample (optional) returning matrices and indices."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if stratify and y is None:
        raise ValueError("To stratify, the 'y' target variable is required.")

    if frac_sample_size >= 1.0:
        return X, X[:0], indices, indices[:0]

    X_sample, X_out_sample, sample_index, out_sample_index = train_test_split(
        X, indices,
        train_size=frac_sample_size,
        random_state=random_state,
        stratify=y if stratify else None,
        shuffle=True
    )

    return X_sample, X_out_sample, sample_index, out_sample_index

#####################################################################################################################

def get_covariance_matrix(
    X: np.ndarray, 
    metric: str, 
    p1: Optional[int], 
    d1: Optional[str], 
    robust_method: str, 
    alpha: float, 
    weights: Optional[list]
) -> Optional[np.ndarray]:
    """Computes the standard or robust covariance matrix if the metric requires it."""
    requires_cov = metric in ['mahalanobis', 'robust_mahalanobis'] or \
                   (metric in MIXED_METRICS and d1 in ['mahalanobis', 'robust_mahalanobis'])

    if not requires_cov or p1 is None or p1 <= 0:
        return None

    X_quant = X[:, :p1]
    
    if 'robust' in metric or 'robust' in str(d1):
        return S_robust(X=X_quant, method=robust_method, alpha=alpha, weights=weights)
    
    return np.cov(X_quant, rowvar=False)

#####################################################################################################################

def get_range(
    X: np.ndarray, 
    metric: str, 
    p1: int, 
):
    
    requires_range = metric == 'simple_gower'

    if not requires_range or p1 is None or p1 <= 0:
        return None
    
    X_quant = X[:,0:p1] 

    return np.max(X_quant, axis=0) - np.min(X_quant, axis=0)


#####################################################################################################################
# 3. ESTIMATOR CLASS (Scikit-Learn API Compatible)
#####################################################################################################################

class SampleDistClustering(BaseEstimator, ClusterMixin):
    """
    Clustering based on subsampling and distance metrics.

    -------------------
    Constructor method
    -------------------

    Parameters: (inputs)
    -----------

    clustering_method: the base clustering algorithm instance (e.g., a scikit-learn or kmedoids object) to be fitted on the subsample's distance matrix.

    metric: the global distance metric to be computed. Must be an string (e.g., 'minkowski', 'robust_mahalanobis', or a mixed metric name).

    frac_sample_size: the sample size in proportional terms to be extracted for the initial clustering. Must be a float in (0, 1].

    random_state: the random seed used for extracting the (random) sample elements.

    stratify: whether to use stratified sampling based on the response variable `y`. Must be a boolean.

    p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be non-negative integers.

    d1: name of the distance to be computed for quantitative variables.

    d2: name of the distance to be computed for binary variables.

    d3: name of the distance to be computed for multi-class variables.

    q: the parameter that defines the Minkowski distance. Must be a positive integer.

    robust_method: the method to be used for computing the robust covariance matrix. Only needed when metric or d1 = 'robust_mahalanobis'.

    alpha: a real number in [0,1] used by the robust covariance estimation method. Only needed when metric or d1 = 'robust_mahalanobis'.

    -----------
    Fit method: 
    -----------

    Fits the subsample-based clustering algorithm to `X`, and `y` (if stratification is required).

    Parameters: (inputs)
    -----------

    X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix and is required. 
    If using mixed metrics, the first p1 predictors must be the quantitative, followed by the p2 binary predictors, and finally the p3 multiclass predictors.

    y: a pandas/polars series or a numpy array. Represents a response variable. Only required if `stratify=True`.

    weights: the sample weights. Used internally for robust covariance estimation if metric or d1 = 'robust_mahalanobis'. 

    ---------------
    Predict method:
    ---------------

    Predicts clusters for `X` observations by assigning them to their nearest medoid (found during the fit stage) according to the configured metric.

    Parameters: (inputs)
    -----------

    X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix and is required. 
    Must follow the same column structure as the `X` passed to the fit method.

    Returns: (outputs)
    --------

    predicted_labels: a list containing the predicted clusters of each observation of `X`.
    """
    def __init__(
        self,
        clustering_method: Any,
        metric: str,
        frac_sample_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        p3: Optional[int] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
        d3: Optional[str] = None,
        q: int = 1,
        robust_method: str = 'trimmed',
        alpha: float = 0.05
    ):
        # In scikit-learn, __init__ MUST ONLY assign arguments to self.
        self.clustering_method = clustering_method
        self.metric = metric
        self.frac_sample_size = frac_sample_size
        self.random_state = random_state
        self.stratify = stratify
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.q = q
        self.robust_method = robust_method
        self.alpha = alpha

    def _compute_dist_matrix(self, X: np.ndarray) -> Union[np.ndarray, Tuple]:
            """Computes the distance matrix based on the estimator's configuration."""
            func = DIST_MATRIX_FUNCS.get(self.metric)
            if not func:
                raise ValueError(f"Metric '{self.metric}' is not supported.")

            if self.metric == 'minkowski':
                D = func(X, self.q)
            
            elif self.metric == 'robust_mahalanobis':
                # Assumes S_robust_est should be computed internally or use the general one
                S_est = S_robust(X, method=self.robust_method, alpha=self.alpha, weights=self.weights)
                D = func(X, S_est)

            elif self.metric == 'simple_gower':
                D = func(X, self.p1, self.p2, self.p3)
            
            elif self.metric in MIXED_METRICS:
                D, D1, D2, D3 = func(
                    X, self.p1, self.p2, self.p3, self.d1, self.d2, self.d3,
                    q=self.q, robust_method=self.robust_method, alpha=self.alpha, 
                    weights=self.weights, return_combined_distances=True
                )
                return D, D1, D2, D3
            
            else:
                D = func(X)

            return D

    def _compute_point_dist(self, xi: np.ndarray, xr: np.ndarray) -> float:
        """Computes the distance between two individual vectors."""
        func = DIST_FUNCS.get(self.metric if self.metric not in MIXED_METRICS else 'ggower')
        
        if self.metric == 'minkowski':
            d = func(xi, xr, self.q)

        elif self.metric in ['mahalanobis', 'robust_mahalanobis']:
            d = func(xi, xr, self.S_estimation_)

        elif self.metric == 'simple_gower':
            D = func(xi, xr, self.range_estimation_, self.p1, self.p2, self.p3)

        elif self.metric in MIXED_METRICS:
            d = func(
                xi, xr, p1=self.p1, p2=self.p2, p3=self.p3, d1=self.d1, d2=self.d2, d3=self.d3,
                q=self.q, S=self.S_estimation_, 
                geom_var_1=self.geom_var_1_, geom_var_2=self.geom_var_2_, geom_var_3=self.geom_var_3_
            )
        else:
            d = func(xi, xr)
        return d

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Union[np.ndarray, Any]] = None, weights: Optional[list] = None):
        """Fits the model to the data."""
        # Safely and uniformly converts Polars/Pandas to Numpy
        X = check_array(X, accept_sparse=False, dtype=None)
        if y is not None:
            y = check_array(y, ensure_2d=False, dtype=None)

        X_sample, X_out_sample, self.sample_idx, out_sample_idx = extract_sample(
            X, y, self.frac_sample_size, self.random_state, self.stratify
        )
        
        self.weights = weights

        if weights is not None:
            self.weights = weights[self.sample_idx] 

        # 1. Distance computation
        self.dist_output = self._compute_dist_matrix(X_sample)
        
        if self.metric in MIXED_METRICS:
            D, D1, D2, D3 = self.dist_output
            self.geom_var_1_ = geometric_variability(D1**2)
            self.geom_var_2_ = geometric_variability(D2**2)
            self.geom_var_3_ = geometric_variability(D3**2)
        else:
            D = self.dist_output
            self.geom_var_1_ = self.geom_var_2_ = self.geom_var_3_ = None

        # 2. Base clustering fit
        self.clustering_method.fit(D)

        # 3. Medoid and Label extraction
        if hasattr(self.clustering_method, 'labels_'): # scikit-learn
            sample_labels = self.clustering_method.labels_
            self.medoid_indices_ = self.clustering_method.medoid_indices_
        else: # kmedoids library
            sample_labels = self.clustering_method.labels
            self.medoid_indices_ = self.clustering_method.medoids

        self.medoids_idx_labels_ = {i: sample_labels[i] for i in self.medoid_indices_}

        # FIX: Extract and save the actual medoid vectors from X_sample!
        # This completely avoids indexing errors later and saves memory.
        self.medoid_vectors_ = X_sample[self.medoid_indices_, :]

        # 4. Covariance Matrix
        self.S_estimation_ = get_covariance_matrix(
            X_sample, self.metric, self.p1, self.d1, self.robust_method, self.alpha, self.weights
        )

        # 6. Range (for simple gower)
        self.range_estimation_ = get_range(
            X_sample, self.metric, self.p1
        )

        # 7. Predict remaining sample
        out_sample_labels = self.predict(X_out_sample) if X_out_sample.shape[0] > 0 else np.array([])
        
        # 8. Reconstruct ordered labels
        data_idx_shuffled = np.concatenate([self.sample_idx, out_sample_idx])
        labels_shuffled = np.concatenate([sample_labels, out_sample_labels])
        
        self.labels_ = labels_shuffled[np.argsort(data_idx_shuffled)]

        return self

    def predict(self, X: Union[np.ndarray, Any]) -> List[int]:
        """Assigns labels to new data based on the nearest medoid."""
        check_is_fitted(self, ['medoid_indices_', 'medoid_vectors_'])
        X = check_array(X, accept_sparse=False, dtype=None)
        
        predicted_labels = []
        for x_new in X:
            # Iterate directly over the saved medoid vectors
            distances = [self._compute_point_dist(x_new, mv) for mv in self.medoid_vectors_]
            # Find the position of the minimum distance (0, 1, 2, ...)
            nearest_idx_in_array = np.argmin(distances)
            
            # Retrieve the original label assigned to that medoid position
            original_medoid_idx = self.medoid_indices_[nearest_idx_in_array]
            predicted_labels.append(self.medoids_idx_labels_[original_medoid_idx])

        return predicted_labels

#####################################################################################################################

class FoldSampleDistClustering(BaseEstimator, ClusterMixin):
    """
    K-Fold Meta-Clustering based on subsampling and distance metrics.
    Implements a two-stage ensemble algorithm relying on SampleDistClustering:
    1. Splits data into K folds and fits a SampleDistClustering model locally on each.
    2. Pools all local medoids and fits a global SampleDistClustering model to find meta-medoids.
    3. Reconstructs the final labels mapping back to the original points.
    """
    def __init__(
        self,
        clustering_method: Any,
        metric: str,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: bool = False,
        frac_sample_size: float = 0.1,
        meta_frac_sample_size: float = 0.8, 
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        p3: Optional[int] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
        d3: Optional[str] = None,
        q: int = 1,
        robust_method: str = 'trimmed',
        alpha: float = 0.05,
    ):
        self.clustering_method = clustering_method
        self.metric = metric
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.frac_sample_size = frac_sample_size
        self.meta_frac_sample_size = meta_frac_sample_size
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.q = q
        self.robust_method = robust_method
        self.alpha = alpha

    def _build_base_model(self, frac_size: float, random_seed: int, stratify_flag: bool) -> 'SampleDistClustering':
        """Helper method to instantiate the underlying clustering component."""
        return SampleDistClustering(
            clustering_method=clone(self.clustering_method),
            metric=self.metric,
            frac_sample_size=frac_size,
            random_state=random_seed,
            stratify=stratify_flag,
            p1=self.p1, p2=self.p2, p3=self.p3,
            d1=self.d1, d2=self.d2, d3=self.d3,
            q=self.q, robust_method=self.robust_method, alpha=self.alpha
        )

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Union[np.ndarray, Any]] = None, weights: Optional[list] = None):
        """Fits the 2-stage meta-clustering model delegating to SampleDistClustering."""
        # Convert inputs safely (assuming check_array is imported)
        X = check_array(X, accept_sparse=False, dtype=None)
        if y is not None:
            y = check_array(y, ensure_2d=False, dtype=None)
                    
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        fold_indices = {}
        fold_labels_dict = {}
        all_medoids_list = []
        medoid_to_fold_cluster_map = [] # To trace back pooled medoids to their origin
        
        # ==========================================
        # STAGE 1: LOCAL CLUSTERING (PER FOLD)
        # ==========================================
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            fold_indices[fold_idx] = test_index
            X_fold = X[test_index]
            y_fold = y[test_index] if y is not None else None
            w_fold = weights[test_index] if weights is not None else None
            
            # Delegate entirely to the base class
            local_model = self._build_base_model(self.frac_sample_size, self.random_state + fold_idx, self.stratify)
            local_model.fit(X_fold, y=y_fold, weights=w_fold)

            # Store predictions for the whole fold
            fold_labels_dict[fold_idx] = local_model.labels_
            
            # Pool medoids and track their origin
            local_medoid_labels = list(local_model.medoids_idx_labels_.values())
            for idx, m_vec in enumerate(local_model.medoid_vectors_):
                all_medoids_list.append(m_vec)
                medoid_to_fold_cluster_map.append((fold_idx, local_medoid_labels[idx]))

        # ==========================================
        # STAGE 2: META-CLUSTERING (ALL MEDOIDS)
        # ==========================================
        X_medoids = np.array(all_medoids_list)
        
        # We also delegate Stage 2!
        self.meta_model_ = self._build_base_model(self.meta_frac_sample_size, self.random_state, stratify_flag=False)
        self.meta_model_.fit(X_medoids, y=None, weights=None)
        
        # Store final vectors to comply with checks
        self.medoid_vectors_ = self.meta_model_.medoid_vectors_
        
        # ==========================================
        # STAGE 3: LABEL RECONSTRUCTION MAPPING
        # ==========================================
        final_labels = np.full(len(X), -1)
        
        for meta_idx, meta_label in enumerate(self.meta_model_.labels_):
            # 1. Identify where this medoid came from
            fold_idx, local_cluster_lbl = medoid_to_fold_cluster_map[meta_idx]
            
            # 2. Find all original points in that fold that belonged to that local cluster
            mask = (fold_labels_dict[fold_idx] == local_cluster_lbl)
            original_indices = fold_indices[fold_idx][mask]
            
            # 3. Assign them the global meta-label
            final_labels[original_indices] = meta_label
                
        self.labels_ = final_labels
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> List[int]:
        """Assigns labels to new data delegating to the meta-model."""
        # check_is_fitted(self, ['meta_model_'])
        
        # Since the meta_model is a SampleDistClustering fitted on the pooled medoids,
        # its own predict method does exactly what we need for out-of-sample data!
        return self.meta_model_.predict(X)
    
#####################################################################################################################      