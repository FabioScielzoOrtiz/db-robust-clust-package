import numpy as np
#import polars as pl
#from PyMachineLearning.preprocessing import Encoder, Imputer
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer

#####################################################################################################################

def outlier_contamination(X, col_name, prop_below=0.05, prop_above=None, sigma=2, random_state=123) :
    """
    Contaminates with outliers a data matrix.

    Parameters (inputs)
    ----------
    X: a pandas/polars series. It represents a statistical variable.
    col: the name of a column of `X`.
    prop_below: proportion of outliers generated in the below part of `X`. Only used if below = True.
    prop_above: proportion of outliers generated in the above part of `X`. Only used if above = True.
    sigma: parameter that controls the upper bound of the generated above outliers and the lower bound of the lower outliers.
    random_state: controls the random seed of the random elements.

    Returns (outputs)
    -------
    X_new: the resulting variable after the outlier contamination of `X`.
    outlier_idx_below: the index of the below outliers.
    outlier_idx_above: the index of the above outliers.
    """

    X_new = X.copy()
    Q25 = X_new[col_name].quantile(0.25)
    Q75 = X_new[col_name].quantile(0.75)
    IQR = Q75 - Q25
    lower_bound = Q25 - 1.5*IQR
    upper_bound = Q75 + 1.5*IQR
    np.random.seed(random_state)

    if prop_below is not None:

        n_outliers_below = int(len(X_new)*prop_below)
        outlier_idx_below = np.random.choice(len(X_new), size=n_outliers_below, replace=False)
        outliers_below = np.random.uniform(lower_bound - sigma*np.abs(lower_bound), lower_bound, size=n_outliers_below)
        X_new.loc[outlier_idx_below, col_name] = outliers_below
        return X_new, outlier_idx_below
        

    elif prop_above is not None: 

        n_outliers_above = int(len(X_new)*prop_above)
        outlier_idx_above = np.random.choice(len(X_new), size=n_outliers_above, replace=False)
        outliers_above = np.random.uniform(upper_bound, upper_bound + sigma*np.abs(upper_bound), size=n_outliers_above)
        X_new.loc[outlier_idx_above, col_name] = outliers_above
        return X_new, outlier_idx_above
    
    elif prop_below is not None and prop_above is not None:

        n_outliers_below = int(len(X_new)*prop_below)
        outlier_idx_below = np.random.choice(len(X_new), size=n_outliers_below, replace=False)
        outliers_below = np.random.uniform(lower_bound - sigma*np.abs(lower_bound), lower_bound, size=n_outliers_below)
        X_new.loc[outlier_idx_below, col_name] = outliers_below

        n_outliers_above = int(len(X_new)*prop_above)
        outlier_idx_above = np.random.choice(len(X_new), size=n_outliers_above, replace=False)
        outliers_above = np.random.uniform(upper_bound, upper_bound + sigma*np.abs(upper_bound), size=n_outliers_above)
        X_new.loc[outlier_idx_above, col_name] = outliers_above

        return X_new, outlier_idx_below, outlier_idx_above

    else:
        raise ValueError('prop_below and prop_above cannot be both None.')


#####################################################################################################################

'''
def sort_predictors_for_GGower(df, quant_predictors, cat_predictors):
    """
    Given a data-frame th function return the names of its categorical variables sorted according to (binary, multi-class) 
    and the number of quantitative, binary and multi-class variables.

    Parameters (inputs)
    ----------
    df: a pandas/polars data-frame. It represents a data matrix.
    quant_predictors: a list with the names of the quantitative variables of `df`.
    cat_predictors: a list with the names of the categorical variables of `df`.

    Returns (outputs)
    -------
    cat_predictors_sorted: a list with the names of the categorical variables of `df` sorted according to (binary, multi-class).
    p1, p2, p3: the number of quantitative, binary and multi-class variables in `df`, respectively.
    """

    # Defining the transformers pipeline to impute and codify the predictors that need it.
    quant_pipeline = Pipeline([
    ('imputer', Imputer(method='simple_mean'))
    ])

    cat_pipeline = Pipeline([
        ('encoder', Encoder(method='ordinal')), # encoding the categorical variables is needed by some imputers
        ('imputer', Imputer(method='simple_most_frequent'))
        ])

    quant_cat_transformer = ColumnTransformer(transformers=[('quant', quant_pipeline, quant_predictors),
                                                            ('cat', cat_pipeline, cat_predictors)])

    predictors = quant_predictors + cat_predictors
    if isinstance(df, pl.DataFrame):
        X = df[predictors].to_pandas()
        # The Null values of the Polars columns that are define as Object type by Pandas are treated as None and not as NaN (what we would like)
        # The avoid this behavior the next step is necessary
        X = X.fillna(value=np.nan)
    # First we have to impute missing values so that are not detected as another unique value
    X = pl.DataFrame(quant_cat_transformer.fit_transform(X))
    X.columns = quant_predictors + cat_predictors
    # Compute number of unique values for each categorical predictor
    n_unique_val = {}
    for col in cat_predictors:
        n_unique_val[col] = len(X[col].unique())
    # Define the list of binary and multi-class predictors based on the number of unique values.
    binary_predictors = [col for col in n_unique_val.keys() if n_unique_val[col] == 2]
    multiclass_predictors = [col for col in n_unique_val.keys() if n_unique_val[col] >= 3]
    # Reorder the list of categorical predictors in a suitable order for Gower Generalized
    cat_predictors_sorted = binary_predictors + multiclass_predictors
    # Getting the number of quant, binary and multi-class predictors
    p1 = len(quant_predictors)
    p2 = len(binary_predictors)
    p3 = len(multiclass_predictors)

    return cat_predictors_sorted, p1, p2, p3
'''