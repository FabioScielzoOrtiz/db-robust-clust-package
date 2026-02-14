#####################################################################################################################

import polars as pl
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids, CLARA
from kmedoids import KMedoidsResult

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
# 3. ESTIMATOR CLASS (Scikit-Learn API Compatible)
#####################################################################################################################

def compute_dist_matrix(
        X, 
        metric, 
        p1=None, 
        p2=None, 
        p3=None, 
        d1=None, 
        d2=None, 
        d3=None,
        q=None, 
        robust_method=None, 
        alpha=None, 
        weights=None,
):

    dist_matrix_objects = DIST_MATRIX_FUNCS
    
    try:

        if metric == 'minkowski':
            D = dist_matrix_objects[metric](X, q)
        elif metric == 'robust_mahalanobis':
            S_robust_est = S_robust(X, method=robust_method, alpha=alpha, weights=weights)
            D = dist_matrix_objects[metric](X, S_robust_est)
        elif metric == 'ggower':
            D, D1, D2, D3 = dist_matrix_objects[metric](
                X, 
                p1, p2, p3,
                d1, d2, d3, 
                q=q, robust_method=robust_method, alpha=alpha, weights=weights,
                return_combined_distances = True
            )
        elif metric == 'relms':
            D, D1, D2, D3 = dist_matrix_objects[metric](
                X, 
                p1, p2, p3,
                d1, d2, d3, 
                q=q, robust_method=robust_method, alpha=alpha, weights=weights,
                return_combined_distances = True
            )
        else: 
            D = dist_matrix_objects[metric](X)

    except Exception as e: 
        raise e
    
    if metric in MIXED_METRICS:
        return D, D1, D2, D3
    
    return D

#####################################################################################################################

def compute_dist(
        xi, 
        xr,
        metric, 
        p1=None, 
        p2=None, 
        p3=None, 
        d1=None, 
        d2=None, 
        d3=None,
        geom_var_1=None,
        geom_var_2=None,
        geom_var_3=None,
        q=None, 
        S=None
    ):

    dist_objects = DIST_FUNCS
    
    if metric == 'minkowski':
        d = dist_objects[metric](xi, xr, q)
    elif metric == 'mahalanobis':
        d = dist_objects[metric](xi, xr, S)
    elif metric == 'robust_mahalanobis':
        d = dist_objects[metric](xi, xr, S)
    elif metric in MIXED_METRICS: 
        d = dist_objects['ggower']( 
            xi, xr, p1=p1, p2=p2, p3=p3, d1=d1, d2=d2, d3=d3, 
            q=q, S=S, geom_var_1=geom_var_1, geom_var_2=geom_var_2, geom_var_3=geom_var_3)
    else: 
        d = dist_objects[metric](xi, xr)

    return d

#####################################################################################################################

def get_covariance_matrix(X, metric, p1, d1, robust_method=None, alpha=None, weights=None):
    
    if metric in ['mahalanobis', 'robust_mahalanobis'] or (metric in MIXED_METRICS and d1 in ['mahalanobis', 'robust_mahalanobis']):

        if p1 != None and p1 > 0:
            X_quant = X[:, :p1] 
            
            if metric == 'robust_mahalanobis' or d1 == 'robust_mahalanobis':
                S = S_robust(X=X_quant, method=robust_method, alpha=alpha, weights=weights)
            
            if metric == 'mahalanobis' or d1 == 'mahalanobis':
                S = np.cov(X_quant, rowvar=False)

            return S

#####################################################################################################################

def get_medoids_idx_labels(sample_labels, medoid_indices):
    
    medoids_idx_labels = {i: sample_labels[i] for i in medoid_indices}

    return medoids_idx_labels

#####################################################################################################################

def get_nearest_medoid_idx(x_new, X, medoid_indices, metric, p1, p2, p3, d1, d2, d3, q, geom_var_1, geom_var_2, geom_var_3, S):

    nearest_medoid_idx = medoid_indices[
        np.argmin([
            compute_dist(
                x_new, X[medoid_idx,:], 
                metric=metric, p1=p1, p2=p2, p3=p3, 
                d1=d1, d2=d2, d3=d3, q=q, 
                geom_var_1=geom_var_1, geom_var_2=geom_var_2, geom_var_3=geom_var_3, S=S
            ) 
            for medoid_idx in medoid_indices
        ])
    ]

    return nearest_medoid_idx

#####################################################################################################################

def get_out_sample_labels(X_sample, X_out_sample, medoid_indices, medoids_idx_labels, metric, p1, p2, p3, d1, d2, d3, q, geom_var_1, geom_var_2, geom_var_3, S):
   
    out_sample_labels = []


    for i in range(len(X_out_sample)):
     
        nearest_medoid_idx = get_nearest_medoid_idx(X_out_sample[i,:], X_sample, medoid_indices, metric, p1, p2, p3, d1, d2, d3, q, geom_var_1, geom_var_2, geom_var_3, S)
    
        out_sample_labels.append(medoids_idx_labels[nearest_medoid_idx])
    
    return out_sample_labels

#####################################################################################################################

def get_labels(sample_index, out_sample_index, sample_labels, out_sample_labels):

    data_idx_shuffled = np.concatenate([sample_index, out_sample_index])
    labels_shuffled = np.concatenate([sample_labels, out_sample_labels])
    data_idx_shuffled_argsort = np.argsort(data_idx_shuffled)
    labels = np.array([labels_shuffled[i]   for i in data_idx_shuffled_argsort])

    return labels

#####################################################################################################################

class SampleDistClustering(BaseEstimator, ClusterMixin):
    """
    Clustering based on subsampling and distance metrics.
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
        alpha: float = 0.05,
        weights: Optional[list] = None
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
        self.weights = weights

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
        elif self.metric in MIXED_METRICS:
            d = func(
                xi, xr, p1=self.p1, p2=self.p2, p3=self.p3, d1=self.d1, d2=self.d2, d3=self.d3,
                q=self.q, S=self.S_estimation_, 
                geom_var_1=self.geom_var_1_, geom_var_2=self.geom_var_2_, geom_var_3=self.geom_var_3_
            )
        else:
            d = func(xi, xr)
        return d

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Union[np.ndarray, Any]] = None):
        """Fits the model to the data."""
        # Safely and uniformly converts Polars/Pandas to Numpy
        X = check_array(X, accept_sparse=False, dtype=None)
        if y is not None:
            y = check_array(y, ensure_2d=False, dtype=None)

        X_sample, X_out_sample, sample_idx, out_sample_idx = extract_sample(
            X, y, self.frac_sample_size, self.random_state, self.stratify
        )

        # 1. Distance computation
        dist_output = self._compute_dist_matrix(X_sample)
        
        if self.metric in MIXED_METRICS:
            D, D1, D2, D3 = dist_output
            self.geom_var_1_ = geometric_variability(D1**2)
            self.geom_var_2_ = geometric_variability(D2**2)
            self.geom_var_3_ = geometric_variability(D3**2)
        else:
            D = dist_output
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

        # 5. Predict remaining sample
        out_sample_labels = self.predict(X_out_sample)
        
        # 6. Reconstruct ordered labels
        data_idx_shuffled = np.concatenate([sample_idx, out_sample_idx])
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
    Implements a two-stage ensemble algorithm:
    1. Splits data into K folds and finds local medoids for each partition.
    2. Pools all local medoids and clusters them again to find the global meta-medoids.
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
        meta_frac_sample_size: float = 0.8, # Replicates the 0.80 used in the meta-clustering step
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        p3: Optional[int] = None,
        d1: Optional[str] = None,
        d2: Optional[str] = None,
        d3: Optional[str] = None,
        q: int = 1,
        robust_method: str = 'trimmed',
        alpha: float = 0.05,
        weights: Optional[list] = None
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
        self.weights = weights

    def _compute_dist_matrix(self, X: np.ndarray) -> Union[np.ndarray, Tuple]:
        """Computes the distance matrix based on the estimator's configuration."""
        func = DIST_MATRIX_FUNCS.get(self.metric)
        if not func:
            raise ValueError(f"Metric '{self.metric}' is not supported.")

        if self.metric == 'minkowski':
            return func(X, self.q)
        elif self.metric in ['mahalanobis', 'robust_mahalanobis']:
            return func(X, self.S_estimation_)
        elif self.metric in MIXED_METRICS:
            return func(
                X, self.p1, self.p2, self.p3, self.d1, self.d2, self.d3,
                q=self.q, robust_method=self.robust_method, alpha=self.alpha, 
                weights=self.weights, return_combined_distances=True
            )
        return func(X)

    def _compute_point_dist(self, xi: np.ndarray, xr: np.ndarray) -> float:
        """Computes the distance between two individual vectors."""
        func = DIST_FUNCS.get(self.metric if self.metric not in MIXED_METRICS else 'ggower')
        
        if self.metric == 'minkowski':
            return func(xi, xr, self.q)
        elif self.metric in ['mahalanobis', 'robust_mahalanobis']:
            return func(xi, xr, self.S_estimation_)
        elif self.metric in MIXED_METRICS:
            return func(
                xi, xr, p1=self.p1, p2=self.p2, p3=self.p3, d1=self.d1, d2=self.d2, d3=self.d3,
                q=self.q, S=self.S_estimation_, 
                geom_var_1=self.geom_var_1_, geom_var_2=self.geom_var_2_, geom_var_3=self.geom_var_3_
            )
        return func(xi, xr)

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Union[np.ndarray, Any]] = None):
        """Fits the 2-stage meta-clustering model."""
        X = check_array(X, accept_sparse=False, dtype=None)
        
        # 0. Global Covariance (Calculated once to ensure scale consistency across all folds)
        self.S_estimation_ = get_covariance_matrix(
            X, self.metric, self.p1, self.d1, self.robust_method, self.alpha, self.weights
        )

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        fold_indices = {}
        fold_labels_dict = {}
        all_medoids_list = []
        
        # ==========================================
        # STAGE 1: LOCAL CLUSTERING (PER FOLD)
        # ==========================================
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            fold_indices[fold_idx] = test_index
            X_fold = X[test_index]
            y_fold = y[test_index] if y is not None else None
            
            X_sample, X_out_sample, sample_idx, out_sample_idx = extract_sample(
                X_fold, y=y_fold, frac_sample_size=self.frac_sample_size, 
                random_state=self.random_state + fold_idx, stratify=self.stratify
            )
            
            dist_output = self._compute_dist_matrix(X_sample)
            if self.metric in MIXED_METRICS:
                D_fold, D1, D2, D3 = dist_output
                self.geom_var_1_ = geometric_variability(D1**2)
                self.geom_var_2_ = geometric_variability(D2**2)
                self.geom_var_3_ = geometric_variability(D3**2)
            else:
                D_fold = dist_output
                self.geom_var_1_ = self.geom_var_2_ = self.geom_var_3_ = None

            # Clone base estimator so it starts fresh for each fold
            fold_clusterer = clone(self.clustering_method)
            fold_clusterer.fit(D_fold)

            if hasattr(fold_clusterer, 'labels_'):
                sample_labels = fold_clusterer.labels_
                medoid_indices = fold_clusterer.medoid_indices_
            else:
                sample_labels = fold_clusterer.labels
                medoid_indices = fold_clusterer.medoids

            # Order medoids logically by cluster label (0, 1, 2...) for consistency
            medoids_idx_labels = {i: sample_labels[i] for i in medoid_indices}
            label_to_medoid_idx = {lbl: idx for idx, lbl in medoids_idx_labels.items()}
            
            fold_medoid_vectors = []
            for lbl in sorted(label_to_medoid_idx.keys()):
                m_idx = label_to_medoid_idx[lbl]
                fold_medoid_vectors.append(X_sample[m_idx])
                all_medoids_list.append(X_sample[m_idx]) # Pool for meta-clustering
                
            # Predict out-of-sample within the fold
            out_sample_labels = []
            for x_out in X_out_sample:
                distances = [self._compute_point_dist(x_out, mv) for mv in fold_medoid_vectors]
                out_sample_labels.append(np.argmin(distances)) # The index IS the label
                
            # Reconstruct full fold labels
            data_idx_shuffled = np.concatenate([sample_idx, out_sample_idx])
            labels_shuffled = np.concatenate([sample_labels, out_sample_labels])
            fold_labels_dict[fold_idx] = labels_shuffled[np.argsort(data_idx_shuffled)]


        # ==========================================
        # STAGE 2: META-CLUSTERING (ALL MEDOIDS)
        # ==========================================
        X_medoids = np.array(all_medoids_list)
        
        X_meta_sample, X_meta_out, meta_sample_idx, meta_out_idx = extract_sample(
            X_medoids, y=None, frac_sample_size=self.meta_frac_sample_size, 
            random_state=self.random_state, stratify=False
        )

        dist_output_meta = self._compute_dist_matrix(X_meta_sample)
        if self.metric in MIXED_METRICS:
            D_meta, D1, D2, D3 = dist_output_meta
            self.geom_var_1_ = geometric_variability(D1**2)
            self.geom_var_2_ = geometric_variability(D2**2)
            self.geom_var_3_ = geometric_variability(D3**2)
        else:
            D_meta = dist_output_meta
            
        meta_clusterer = clone(self.clustering_method)
        meta_clusterer.fit(D_meta)
        
        if hasattr(meta_clusterer, 'labels_'):
            meta_sample_labels = meta_clusterer.labels_
            meta_medoid_indices = meta_clusterer.medoid_indices_
        else:
            meta_sample_labels = meta_clusterer.labels
            meta_medoid_indices = meta_clusterer.medoids
            
        # SAVING THE FINAL VECTORS! 
        self.medoid_vectors_ = X_meta_sample[meta_medoid_indices, :]
        self.meta_medoids_idx_labels_ = {i: meta_sample_labels[i] for i in meta_medoid_indices}
        
        meta_out_labels = []
        for x_out in X_meta_out:
            distances = [self._compute_point_dist(x_out, mv) for mv in self.medoid_vectors_]
            nearest_idx = np.argmin(distances)
            original_medoid_idx = meta_medoid_indices[nearest_idx]
            meta_out_labels.append(self.meta_medoids_idx_labels_[original_medoid_idx])
            
        meta_idx_shuffled = np.concatenate([meta_sample_idx, meta_out_idx])
        meta_labels_shuffled = np.concatenate([meta_sample_labels, meta_out_labels])
        final_meta_labels = meta_labels_shuffled[np.argsort(meta_idx_shuffled)]
        
        # ==========================================
        # STAGE 3: LABEL RECONSTRUCTION MAPPING
        # ==========================================
        final_labels = np.full(len(X), -1)
        meta_idx = 0
        n_clusters_per_fold = len(medoid_indices)
        
        for fold_idx in range(self.n_splits):
            for cluster_label in range(n_clusters_per_fold):
                # Get the meta-label assigned to this specific fold's cluster
                assigned_meta_label = final_meta_labels[meta_idx]
                
                # Apply the meta-label to all original points that belonged to that fold's cluster
                mask = (fold_labels_dict[fold_idx] == cluster_label)
                original_indices = fold_indices[fold_idx][mask]
                final_labels[original_indices] = assigned_meta_label
                
                meta_idx += 1
                
        self.labels_ = final_labels
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> List[int]:
        """Assigns labels to new data based on the final meta-medoids."""
        check_is_fitted(self, ['medoid_vectors_'])
        X = check_array(X, accept_sparse=False, dtype=None)
        
        predicted_labels = []
        for x_new in X:
            distances = [self._compute_point_dist(x_new, mv) for mv in self.medoid_vectors_]
            nearest_idx_in_array = np.argmin(distances)
            
            original_medoid_idx = list(self.meta_medoids_idx_labels_.keys())[nearest_idx_in_array]
            predicted_labels.append(self.meta_medoids_idx_labels_[original_medoid_idx])

        return predicted_labels
    
#####################################################################################################################      
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

import polars as pl
import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import KFold
from robust_mixed_dist.mixed import FastGGowerDistMatrix, GGowerDist
from tqdm import tqdm

#####################################################################################################################

def concat_X_y(X, y, y_type, p1, p2, p3):
    """
    Concatenating `X`and `y` in a suitable way to be used by the class `FastKmedoidsGG` to be applied in 'supervised' clustering.

    Parameters (inputs)
    ----------
    X: a numpy array. It represents a predictors matrix.
    y: a numpy array. It represents a response/target variable.
    y_type: the type of response variable. Must be in ['quantitative', 'binary', 'multiclass'].
    p1, p2, p3: number of quantitative, binary and multi-class predictors in `X`.

    Returns (outputs)
    -------
    X_y: the result of concatening `X` and `y` in the proper way to be used in `FastKmedoidsGG`
    p1, p2, p3: the updated number of quantitative, binary and multi-class predictors in `X_y`.
    y_idx: the column index in which `y` is located in `X_y`.
    """

    if y_type == 'binary':
        X_y = np.column_stack((X[:,0:p1], y, X[:,(p1+1):]))
        p2 = p2 + 1 # updating p2 since now X contains y and it is binary. 
        y_idx = p1 
    elif y_type == 'multiclass':
        X_y = np.column_stack((X[:,0:p1], X[:,(p1+1):p2], y, X[:,(p2+1):]))
        p3 = p3 + 1 # updating p3 since now X contains y and it is multiclass. 
        y_idx = p2
    elif y_type == 'quantitative':
        X_y = np.column_stack((y, X))
        p1 = p1 + 1 # updating p1 since now X contains y and it is quant. 
        y_idx = 0
    else:
        raise ValueError("Invalid `y` type")
    
    return X_y, p1, p2, p3, y_idx

#####################################################################################################################
    
def get_idx_obs(fold_key, medoid_key, idx_fold, labels_fold):
    # Idx of the observations of fold_key associated to the medoid_key of that fold
    return idx_fold[fold_key][np.where(labels_fold[fold_key] == medoid_key)[0]]

#####################################################################################################################
        
class FastKmedoidsGGower :
    """
    Implements the Fast-K-medoids algorithm based on the Generalized Gower distance.
    """

    def __init__(self, n_clusters, method='pam', init='heuristic', max_iter=100, random_state=123,
                 frac_sample_size=0.1, p1=None, p2=None, p3=None, d1='robust_mahalanobis', d2='jaccard', d3='matching', 
                 robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, q=1,
                 fast_VG=False, VG_sample_size=1000, VG_n_samples=5, y_type=None) :
        """
        Constructor method.
        
        Parameters:
            n_clusters: the number of clusters.
            method: the k-medoids clustering method. Must be in ['pam', 'alternate']. PAM is the classic one, more accurate but slower.
            init: the k-medoids initialization method. Must be in ['heuristic', 'random']. Heuristic is the classic one, smarter burt slower.
            max_iter: the maximum number of iterations run by k-medodis.
            frac_sample_size: the sample size in proportional terms.
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            robust_method: the method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
            epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            n_iters: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            fast_VG: whether the geometric variability estimation will be full (False) or fast (True).
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
            y_type: the type of response variable. Must be in ['quantitative', 'binary', 'multiclass'].
        """        
        self.n_clusters = n_clusters; self.method = method; self.init = init; self.max_iter = max_iter; self.random_state = random_state
        self.frac_sample_size = frac_sample_size; self.p1 = p1; self.p2 = p2; self.p3 = p3; self.d1 = d1; self.d2 = d2; self.d3 = d3; 
        self.robust_method = robust_method; self.alpha = alpha; self.epsilon = epsilon; self.n_iters = n_iters; self.fast_VG = fast_VG; 
        self.VG_sample_size = VG_sample_size; self.VG_n_samples = VG_n_samples; self.q = q ; self.y_type = y_type
        self.kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', method=method, init=init, max_iter=max_iter, random_state=random_state)

    def fit(self, X, y=None, weights=None):
        """
        Fit method: fitting the fast k-medoids algorithm to `X` (and `y` if needed).
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix. Is required.
            y: a pandas/polars series or a numpy array. Represents a response variable. Is not required.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        """
        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()           
        
        self.p1_init = self.p1 ; self.p2_init = self.p2 ; self.p3_init = self.p3  # p1, p2 and p3 when X doesn't contain y. These original p's are needed for the predict method, since what is predicted is X without y.

        if y is not None: 
            X, self.p1, self.p2, self.p3, self.y_idx = concat_X_y(X=X, y=y, y_type=self.y_type, p1=self.p1, p2=self.p2, p3=self.p3)

        fastGG = FastGGowerDistMatrix(frac_sample_size=self.frac_sample_size, random_state=self.random_state, p1=self.p1, p2=self.p2, p3=self.p3, 
                                      d1=self.d1, d2=self.d2, d3=self.d3, robust_method=self.robust_method, alpha=self.alpha, epsilon=self.epsilon, 
                                      n_iters=self.n_iters, fast_VG=self.fast_VG, VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, 
                                      q=self.q, weights=weights)
        
        fastGG.compute(X)

        self.D_GG = fastGG.D_GGower
        self.X_sample = fastGG.X_sample
        self.X_out_sample = fastGG.X_out_sample
        self.sample_index = fastGG.sample_index
        self.out_sample_index = fastGG.out_sample_index
         
        self.kmedoids.fit(self.D_GG)
        sample_labels_dict = {idx : self.kmedoids.labels_[i] for i, idx in enumerate(self.sample_index)} # keys: observation indices. values: cluster labels. Contains only the sample observation indices.
        self.sample_labels = np.array(list(sample_labels_dict.values()))

        self.medoids_ = {}
        medoids_idx = [int(x) for x in self.kmedoids.medoid_indices_]
        for j, idx in enumerate(medoids_idx):
            self.medoids_[j] = self.X_sample[idx,:] 

        sample_weights = weights[self.sample_index] if weights is not None else None

        self.distGG = GGowerDist(p1=self.p1, p2=self.p2, p3=self.p3, d1=self.d1, d2=self.d2, d3=self.d3, q=self.q,
                                 robust_method=self.robust_method, alpha=self.alpha,  epsilon=self.epsilon, 
                                 n_iters=self.n_iters, VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, 
                                 random_state=self.random_state, weights=sample_weights) 
     
        if sample_weights is None:
            self.distGG.fit(X)
        else: # if there are weights we cannot use X when it is too large in n (number of rows), since Xw is n x n, therefore it cannot be computed in that case due to computational problems. To avoid this potential problem instead of using X to fit GG_dist we use the very reduce sample X_sample.
            self.distGG.fit(self.X_sample) 
        # We could use the VG's computed with GG_matrix in GG_dist, rather than making this second estimation. But the current estimation is very fast (less than 1 second) and is equally accurate. So use one or another lead to the same results.

        dist_out_sample_medoids = {idx : [] for idx in self.out_sample_index} # keys: out sample idx, values: distance with respect each medoid.
        for i, idx in enumerate(self.out_sample_index) :
            for j in range(0, self.n_clusters) :
                dist_out_sample_medoids[idx].append(self.distGG.compute(xi=self.X_out_sample[i,:], xr=self.medoids_[j])) 
       
        out_sample_labels_dict = {idx : np.argmin(dist_out_sample_medoids[idx]) for idx in self.out_sample_index} # keys: observation indices. Values: cluster labels. Contains only the out of sample observation indices
        self.out_sample_labels = np.array(list(out_sample_labels_dict.values()))
        sample_labels_dict.update(out_sample_labels_dict)  # Now sample_label_dict contains the labels for each observation index, but without order.
        labels_dict = {idx : sample_labels_dict[idx] for idx in range(0,len(X))}  # keys: observation indices. Values: cluster labels. Contains all the observation indices
        self.labels_ = np.array(list(labels_dict.values()))

        self.X = X
        self.y = y

    def predict(self, X):
        """
        Predict method: predicting clusters for `X` observation by assigning them to their nearest cluster (medoid) according to Generalized Gower distance.

        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix. Is required.
        """

        if self.y: # remove y from the medoids, since in predict method X doesn't contain y.
            for j in range(self.n_clusters):
                self.medoids_[j] = np.delete(self.medoids_[j], self.y_idx)

        distGG = GGowerDist(p1=self.p1_init, p2=self.p2_init, p3=self.p3_init, d1=self.d1, d2=self.d2, d3=self.d3, q=self.q,
                                robust_method=self.robust_method, alpha=self.alpha, epsilon=self.epsilon, n_iters=self.n_iters,
                                VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, random_state=self.random_state) 
            
        distGG.fit(self.X) # self.X is X used during fit method, not necessarily the X parameter passed to the predict method.

        predicted_clusters = []
        for i in range(0, len(X)):
                dist_xi_medoids = [distGG.compute(xi=X[i,:], xr=self.medoids_[j]) for j in range(self.n_clusters)]
                predicted_clusters.append(np.argmin(dist_xi_medoids))

        return predicted_clusters

#####################################################################################################################

class FoldFastKmedoidsGGower:
    """
    Implements the K-Fold Fast-K-medoids algorithm based on the Generalized Gower distance.
    """

    def __init__(self, n_clusters, method='pam', init='heuristic', max_iter=100, random_state=123,
                 frac_sample_size=0.1, p1=None, p2=None, p3=None, d1='robust_mahalanobis', d2='jaccard', d3='matching', 
                 robust_method='trimmed', alpha=0.05, epsilon=0.05, n_iters=20, q=1, fast_VG=False, 
                 VG_sample_size=1000, VG_n_samples=5, n_splits=5, shuffle=True, kfold_random_state=123, y_type=None) :
        """
        Constructor method.
        
        Parameters:
            n_clusters: the number of clusters.
            method: the k-medoids clustering method. Must be in ['pam', 'alternate']. PAM is the classic one, more accurate but slower.
            init: the k-medoids initialization method. Must be in ['heuristic', 'random']. Heuristic is the classic one, smarter burt slower.
            max_iter: the maximum number of iterations run by k-medodis.
            frac_sample_size: the sample size in proportional terms.
            p1, p2, p3: number of quantitative, binary and multi-class variables in the considered data matrix, respectively. Must be a non negative integer.
            d1: name of the distance to be computed for quantitative variables. Must be an string in ['euclidean', 'minkowski', 'canberra', 'mahalanobis', 'robust_mahalanobis']. 
            d2: name of the distance to be computed for binary variables. Must be an string in ['sokal', 'jaccard'].
            d3: name of the distance to be computed for multi-class variables. Must be an string in ['matching'].
            q: the parameter that defines the Minkowski distance. Must be a positive integer.
            robust_method: the method to be used for computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            alpha : a real number in [0,1] that is used if `method` is 'trimmed' or 'winsorized'. Only needed when d1 = 'robust_mahalanobis'.
            epsilon: parameter used by the Delvin algorithm that is used when computing the robust covariance matrix. Only needed when d1 = 'robust_mahalanobis'.
            n_iters: maximum number of iterations used by the Delvin algorithm. Only needed when d1 = 'robust_mahalanobis'.
            fast_VG: whether the geometric variability estimation will be full (False) or fast (True).
            VG_sample_size: sample size to be used to make the estimation of the geometric variability.
            VG_n_samples: number of samples to be used to make the estimation of the geometric variability.
            random_state: the random seed used for the (random) sample elements.
            y_type: the type of response variable. Must be in ['quantitative', 'binary', 'multiclass'].
            n_splits: number of folds to be used.
            shuffle: whether data is shuffled before applying KFold or not, must be in [True, False]. 
            kfold_random_state: the random seed for KFold if shuffle = True.
        """          
        self.n_clusters = n_clusters; self.method = method; self.init = init; self.max_iter = max_iter; self.random_state = random_state
        self.frac_sample_size = frac_sample_size; self.p1 = p1; self.p2 = p2; self.p3 = p3; self.d1 = d1; self.d2 = d2; self.d3 = d3; 
        self.robust_method = robust_method ; self.alpha = alpha; self.epsilon = epsilon; self.n_iters = n_iters; self.fast_VG = fast_VG; 
        self.VG_sample_size = VG_sample_size;  self.VG_n_samples = VG_n_samples; self.q = q; self.n_splits = n_splits; self.shuffle = shuffle; 
        self.kfold_random_state = kfold_random_state; self.y_type = y_type

    def fit(self, X, y=None, weights=None):
        """
        Fit method: fitting the fast k-medoids algorithm to `X` (and `y` if needed).
        
        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix. Is required.
            y: a pandas/polars series or a numpy array. Represents a response variable. Is not required.
            weights: the sample weights. Only used if provided and d1 = 'robust_mahalanobis'.  
        """
        
        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pl.Series)):
            y = y.to_numpy()

        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.kfold_random_state)

        idx_fold = {}
        for j, (train_index, test_index) in enumerate(kfold.split(X)):
            idx_fold[j] = test_index

        medoids_fold, labels_fold = {}, {}
        for j in tqdm(range(0, self.n_splits), desc="Clustering Folds"):

            fold_weights = weights[idx_fold[j]] if weights is not None else None
            y_fold = y[idx_fold[j]] if y is not None else None

            fast_kmedoids = FastKmedoidsGGower(n_clusters=self.n_clusters, method=self.method, init=self.init, max_iter=self.max_iter, 
                                               random_state=self.random_state, frac_sample_size=self.frac_sample_size, 
                                               p1=self.p1, p2=self.p2, p3=self.p3, d1=self.d1, d2=self.d2, d3=self.d3, 
                                               robust_method=self.robust_method, alpha=self.alpha, epsilon=self.epsilon, 
                                               n_iters=self.n_iters, fast_VG=self.fast_VG, VG_sample_size=self.VG_sample_size, 
                                               VG_n_samples=self.VG_n_samples, y_type=self.y_type)
           
            fast_kmedoids.fit(X=X[idx_fold[j],:], y=y_fold, weights=fold_weights) 
           
            medoids_fold[j] = fast_kmedoids.medoids_
            labels_fold[j] = fast_kmedoids.labels_

        if y is not None:
            self.y_idx = fast_kmedoids.y_idx
            self.p1_init = fast_kmedoids.p1_init; self.p2_init = fast_kmedoids.p2_init; self.p3_init = fast_kmedoids.p3_init            

        X_medoids = np.row_stack([np.array(list(medoids_fold[fold_key].values())) for fold_key in range(0, self.n_splits)])

        fast_kmedoids = FastKmedoidsGGower(n_clusters=self.n_clusters, method=self.method, init=self.init, max_iter=self.max_iter, 
                                           random_state=self.random_state, frac_sample_size=0.80, p1=self.p1, p2=self.p2, p3=self.p3,
                                           d1=self.d1, d2=self.d2, d3=self.d3, robust_method=self.robust_method, alpha=self.alpha, 
                                           epsilon=self.epsilon, n_iters=self.n_iters, fast_VG=self.fast_VG, 
                                           VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples)
       
        fast_kmedoids.fit(X=X_medoids)     

        fold_medoid_keys = [(fold_key, medoid_key) for fold_key in range(0, self.n_splits) for medoid_key in range(0, self.n_clusters)]
        labels_dict = dict(zip(fold_medoid_keys, fast_kmedoids.labels_))
        labels_dict = {fold_key: {medoid_key: labels_dict[fold_key, medoid_key] for medoid_key in range(0,self.n_clusters)} for fold_key in range(0,self.n_splits)}

        final_labels = np.repeat(-1, len(X))
        for fold_key in range(0, self.n_splits):
            for medoid_key in range(0, self.n_clusters):
                final_labels[get_idx_obs(fold_key, medoid_key, idx_fold, labels_fold)] = labels_dict[fold_key][medoid_key]

        self.labels_ = final_labels
        self.medoids_ = fast_kmedoids.medoids_
        self.X = X
        self.y = y

    def predict(self, X):
        """
        Predict method: predicting clusters for `X` observation by assigning them to their nearest cluster (medoid) according to Generalized Gower distance.

        Parameters:
            X: a pandas/polars data-frame or a numpy array. Represents a predictors matrix. Is required.
        """

        if self.y is not None: # remove y from the medoids, since in predict method X doesn't contain y.
            for j in range(self.n_clusters):
                self.medoids_[j] = np.delete(self.medoids_[j], self.y_idx)

        distGG = GGowerDist(p1=self.p1_init, p2=self.p2_init, p3=self.p3_init, d1=self.d1, d2=self.d2, d3=self.d3, q=self.q,
                                robust_method=self.robust_method, alpha=self.alpha, epsilon=self.epsilon, n_iters=self.n_iters,
                                VG_sample_size=self.VG_sample_size, VG_n_samples=self.VG_n_samples, random_state=self.random_state) 
           
        distGG.fit(self.X) # self.X is X used during fit method, not necessarily the X parameter passed to the predict method

        predicted_clusters = []
        for i in range(0, len(X)):
                dist_xi_medoids = [distGG.compute(xi=X[i,:], xr=self.medoids_[j]) for j in range(self.n_clusters)]
                predicted_clusters.append(np.argmin(dist_xi_medoids))

        return predicted_clusters
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################