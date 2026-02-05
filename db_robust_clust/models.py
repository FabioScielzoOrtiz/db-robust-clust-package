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