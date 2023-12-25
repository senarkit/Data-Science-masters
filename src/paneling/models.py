"""
Models:\n
This module consists of all the models that will be used in store paneling. Each model is a class that majorly contains functions to fit-transform, selecting best params, and predict for given test data (if applicable).
"""

import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture  # for GMM clustering
from sklearn.model_selection import RandomizedSearchCV
from utils import logger


class Lme():
    """
    Lme serves as an api for linear mixed effects modeling (internally uses statsmodels mixedlm)
    provides an interface  to fit, predict and aso to get formatted version of random and fixed betas
    """

    def __init__(self, train_data, lme_args, track_tag):
        """
        initializes the class attributes
        Parameters
        ----------
        train_data : DataFrame
            pandas dataframe containing train data
        lme_args : dict
            dictionary containing arguments to configure lme run
            y_col           : column name of dependent variable (str)
            group_col       : column name containing grouping variable for random fixed_effects (str)
            fixed_effects   : list of features to be treated as fixed effects
            random_effects  : list of features to be treated as random effects
        track_tag : str
            a string in the following format (to log/track the combination of the invoking job):
            format: "item_panel_rolling" (ex: 4314_2_20-02-2021)
        Attributes
        ----------
        track_tag : str
        y_col : str
        group_col : str
        fixed_effects : list
        random_effects  : list
        train_data : dataframe
        md : statsmodel-model
            initialised model object (will be used for training and predictions by other class functions)
        param_gen_flag : boolean
            indicates if model parameters are already generated and stored (helps avoid redundent function calls)

        """
        try:
            self.track_tag = track_tag
            self.y_col = lme_args["y_col"]
            self.group_col = lme_args["group_col"]
            self.fixed_effects = lme_args["fixed_effects"]
            self.random_effects = lme_args["random_effects"]
            self.train_data = train_data
            self.train_data['INTERCEPT'] = 1

            self.param_gen_flag = 0

            endog = self.train_data[self.y_col]
            exog = self.train_data[self.fixed_effects]
            exog_re = self.train_data[self.random_effects]
            groups = self.train_data[self.group_col]

            self.md = sm.MixedLM(endog=endog,
                                 groups=groups,
                                 exog=exog,
                                 exog_re=exog_re)
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while initializing lme class", '')
            logger.error(e, exc_info=True)

    def fit(self):
        """
        fit is statsmodels_model.fit()
        fits a linear mixed effect model on a predefiend model (class atribute)
        """
        try:
            self.mdf = self.md.fit()
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while fitting lme", '')
            logger.error(e, exc_info=True)

    def gen_params(self):
        """
        gen_params fetches model params and stores in class attributed for further reference
        executes only if attribute param_gen_flag is 0 (1 indicates that parameters already updated in class attributes)
        Attributes
        ----------
        fixed_params : dict
            trained parameters of fixed fixed_effects
        random params: dict
            trained parameters of random_effects 
        """
        try:
            if not self.param_gen_flag:
                self.fixed_params = self.mdf.params
                self.random_params = self.mdf.random_effects
                self.param_gen_flag = 1
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while generating lme params", '')
            logger.error(e, exc_info=True)

    def get_params(self):
        """
        get_params formats the generated params to a consumable format and returns
        Returns
        -------
        fixed_param_df : DataFrame
            DataFrame containing fixed_effect feature wise learnt coefficient
        random_param_df: DataFrame
            DataFrame containing group x random_effect feature wise learnt coefficient
        """
        try:
            self.gen_params()
            # add code for correct formating align with bayesian params
            random_param_df = pd.DataFrame(self.random_params).T.reset_index()
            random_param_df = random_param_df.rename(columns={"index": self.group_col})
            fixed_param_df = pd.DataFrame(self.fixed_params).reset_index()
            fixed_param_df.columns = ["feature", "param"]
            return fixed_param_df, random_param_df
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while getting lme params", '')
            logger.error(e, exc_info=True)

    def make_pred_col(self, col, track_Tag):
        """
        make_pred_col extends a column name with postfix to make a coresponding column name for predictions
        postfix string is taken from utils/names.py 
        purpose of this function is to standardise the column names representing predictions (by avoiding random string constructions)

        Parameters
        ----------
        col : string
            actual column name of feature to be predicted
        track_tag : string
            an identifier used to log/track the info about of the invoking job
        Returns
        -------
        string
            corresponding prediction column name
        """
        try:
            return col + '_PRED'
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while creating Predict col", '')
            logger.error(e, exc_info=True)

    def predict(self, test_data=pd.DataFrame()):
        """
        predict used trained LME parameters from astats models model to generate predictions on the given dataset
        Parameters
        ----------
        test_data : DataFrame, optional 
            predictions performed on train data if this parameter is not specified or emty
        Returns
        -------
        DataFrame
            group x week level actuals and predictions of dependent variable

        """

        try:
            self.gen_params()
            if not test_data.empty:
                data = test_data.copy()
            else:
                data = self.train_data.copy()
            data['intercept'] = 1
            # vectorize this
            pred_yhat = []
            groups_in_order = data[self.group_col].drop_duplicates().to_list()
            for group_id in data[self.group_col].unique():
                group_df = data[data[self.group_col] == group_id]  # change to names.
                group_yhat = np.zeros(len(group_df))
                for fe in self.fixed_effects:
                    coef = self.fixed_params[fe]
                    group_yhat = group_yhat + group_df[fe].values * coef
                for re in self.random_effects:
                    coef = self.random_params[group_id][re]
                    group_yhat = group_yhat + group_df[re].values * coef
                group_pred_df = pd.DataFrame(group_yhat)
                group_pred_df[1] = group_id
                group_pred_df[2] = group_df[self.y_col].values
                # group_pred_df[3] = group_df['WK_END_DT'].values  # change to names.
                pred_yhat.append(group_pred_df)

            pred_yhat_df = pd.concat(pred_yhat)
            pred_yhat_df.columns = [self.make_pred_col(
                self.y_col, self.track_tag), self.group_col, self.y_col]
            # pred_yhat_df.columns = [self.make_pred_col(
            #     self.y_col, self.track_tag), self.group_col, self.y_col, 'WK_END_DT']

            return pred_yhat_df
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while Predicting using LME", '')
            logger.error(e, exc_info=True)


class gmm():
    """
    gmm serves as an api for Gaussian mixture models (internally uses sklearn GaussianMixture)
    provides an interface to fit and get clusters for the given data
    """

    def __init__(self, train_data, gmm_args):
        """
        Initializes the train data and arguments required for GMM modelling.

        Parameters
        ----------
        train_data : pd.DataFrame
            Contains store level data for store-panel mapping
        gmm_args : dict
            Dictionary of all the arguments required for GMM
        """
        
        try:
            self.train_data = train_data
            self.n_components = gmm_args['num_clusters']
            self.covariance_type = gmm_args['covariance_type']
            self.max_iter = gmm_args['max_iter']
            self.init_params = gmm_args['init_params']

            self.model = GaussianMixture(n_components=self.n_components,  # this is the number of clusters
                                         # {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
                                         covariance_type=self.covariance_type,
                                         max_iter=self.max_iter,  # the number of EM iterations to perform. default=100
                                         n_init=1,  # the number of initializations to perform. default = 1
                                         # the method used to initialize the weights, the means and the precisions. {'random' or default='k-means'}
                                         init_params='kmeans',
                                         verbose=0,  # default 0, {0,1,2}
                                         random_state=0  # for reproducibility
                                         )
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while fitting GMM model", '')
            logger.error(e, exc_info=True)

    def fit(self):
        """
        fit is sklearn.mixture.GaussianMuxture.fit()
        fits a GMM model on a predefiend model (class atribute)
        """
        try:
            self.clus = self.model.fit(self.train_data)
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while fitting GMM model", '')
            logger.error(e, exc_info=True)
            

    def predict(self):
        """
        predict is sklearn.mixture.GaussianMuxture.predict(). Gives the Cluster-labels for each store. 

        Returns
        -------
        array
            Cluster-labels for each store. 
        """
        return self.model.predict(self.train_data)

    def get_model_summary(self):
        """
        get_model_summary summarizes the GMM model metrics like weights, convergence etc.

        Returns
        -------
        Dict
            Dictionary that contains all the metrics and summary 
        """
        try:
            out = {}
            out['Weights'] = self.clus.weights_
            out['Means'] = self.clus.means_
            out['Covariances'] = self.clus.covariances_
            out['Precisions'] = self.clus.precisions_
            out['Precisions Cholesky'] = self.clus.precisions_cholesky_
            out['Converged'] = self.clus.converged_
            out['No. of Iterations'] = self.clus.n_iter_
            out['Lower Bound'] = self.clus.lower_bound_
            return out
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while getting GMM model summary", '')
            logger.error(e, exc_info=True)



class rf():
    """
    rf serves as an api for Random Forest models (internally uses sklearn's RandomForestClassifier)
    provides an interface to fit and classifies the stores into existing panels.
    """

    def __init__(self, train_data, test_data, rf_args):
        """
        Initializes the train data and arguments required for RF modelling.

        Parameters
        ----------
        train_data : pd.DataFrame
            Contains store level data for store-panel mapping for old stores
        test_data : pd.DataFrame
            Contains store level data for store-panel mapping for new stores
        rf_args : dict
            Dictionary of all the arguments required for RF
        """
        try:
            self.train_data = train_data
            self.test_data = test_data
            self.y_col = rf_args["y_col"]

            self.X_train = self.train_data.drop(columns=self.y_col)
            self.y_train = self.train_data[self.y_col]

            random_grid = {'n_estimators': rf_args['n_estimators'],                 # Number of trees in random forest
                           # Number of features to consider at every split
                           'max_features': rf_args['max_features'],
                           'max_depth': rf_args['max_depth'],                      # Maximum number of levels in tree
                           # Minimum number of samples required to split a node
                           'min_samples_split': rf_args['min_samples_split'],
                           # Minimum number of samples required at each leaf node
                           'min_samples_leaf': rf_args['min_samples_leaf'],
                           'bootstrap': rf_args['bootstrap'],                      # [True, False]
                           'criterion': ['gini', 'entropy']}

            self.rf = self.fit(random_grid)

        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while initializing RF class", '')
            logger.error(e, exc_info=True)

    def fit(self, random_grid):
        """
        fit is RandomForestClassifier.fit() after getting the best params from RandomizedSearchCV
        fits a RF model on a predefiend model (class atribute)
        Parameters
        ----------
        random_grid : dict
            Dictionary for all arguments required for random search

        """
        try:
            rf = RandomForestClassifier()
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                           n_iter=10, cv=10, verbose=0, random_state=42, n_jobs=8)
            rf_random.fit(self.X_train, self.y_train.tolist())
            return rf_random
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while fitting RF model", '')
            logger.error(e, exc_info=True)

    def predict(self):
        """
        predict is sklearn.mixture.GaussianMuxture.predict(). Gives the Cluster-labels for each store. 

        Returns
        -------
        pd.DataFrame
            Contains final Cluster mapping data
        """
        try:
            # predicted = self.rf.predict(self.test_data)
            all_data = pd.concat([self.X_train, self.test_data])
            all_stores_pred = self.rf.predict(all_data)
            all_data['RF_predict'] = all_stores_pred
            actual_clusters = self.y_train.to_dict()
            all_data['OLD_CLUSTER'] = all_data.index.map(actual_clusters)
            all_data['FINAL_CLUSTER'] = np.where(all_data['OLD_CLUSTER'].isna(),
                                                all_data['RF_predict'], all_data['OLD_CLUSTER'])
            all_data = all_data.reset_index()
            return all_data
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while predicting using RF", '')
            logger.error(e, exc_info=True)
        

    def get_model_summary(self):
        """
        get_model_summary summarizes the GMM model metrics like Feature importance, Best params selected by random search

        Returns
        -------
        Dict
            Dictionary that contains all the metrics and summary 
        """
        try:
            out = {}
            feature_importance = pd.Series(self.rf.best_estimator_.feature_importances_,
                                        index=self.X_train.columns).sort_values(ascending=False)
            feature_importance = pd.DataFrame(self.rf.best_estimator_.feature_importances_, index=self.X_train.columns)
            feature_importance.columns = ['Feature_Importance']
            feature_importance = feature_importance.sort_values(by=['Feature_Importance'], ascending=False)
            # print(feature_importance[:10])
            out['Feature_importance'] = feature_importance
            out['Best_Params'] = self.rf.best_params_
            return out
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while generating RF summary", '')
            logger.error(e, exc_info=True)
