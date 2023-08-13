import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class feature_selection:
    """
    provides an interface to get respective features and their order of importance
    """
    def __init__(self, X, y, features, model_name):
        """
        initializes the class attributes
        Parameters
        -----------
        X : DataFrame
            pandas dataframe with all the features
        y : Series
            numpy series containing the target feature
        feat_list : list
            list containing the list of features to be used
            should contain in X
        model_name : str
            choose which type of model to use for feature selection        
        """
        try:
            self.X = X
            self.y = y
            self.features = features
            self.model_name = model_name

            logger.info("class selection initialized")
        except Exception as e:
            logger.info("%s - failed while initializing selection class", '')
            logger.error(e, exc_info=True)



    def fit_predict(self):
        coef_df = pd.DataFrame(columns=['Features', 'Coeff'])
        X = self.X
        y = self.y

        ## treat categorical features
        X_num = X.select_dtypes(exclude=['object']).fillna(0)
        X_obj = pd.DataFrame()
        for col in X.select_dtypes(include=['object']).columns:
                tmp = pd.get_dummies(X[col], prefix=col)
                X_obj = pd.concat([X_obj, tmp], axis=1)
        X = pd.concat([X_num, X_obj], axis=1)
        
        ## Scale data
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        y_scaled = ss.fit_transform(np.array(y).reshape(-1,1))
        y_scaled = y_scaled.flatten()

        
        ## Fit Model
        if self.model_name.lower() == 'linear_regressor':
            model = LinearRegression()
            model.fit(X_scaled, y_scaled)
            coef_df['Features'] = X_scaled.columns
            coef_df['Coeff'] = model.coef_
        
        if self.model_name.lower() == 'logistic_regressor':
            model = LogisticRegression()
            # Encode target variable
            y_scaled[abs(y_scaled) < 0.5] = 0
            y_scaled[abs(y_scaled) > 0.5] = 1
            model.fit(X_scaled, y_scaled)
            coef_df['Features'] = X_scaled.columns
            coef_df['Coeff'] = model.coef_[0]

        logger.info("fit_predict executed successfully")
        return coef_df