import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
        try:
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
            # condition below step only for regression problem
            y_scaled = ss.fit_transform(np.array(y).reshape(-1,1))
            y_scaled = y_scaled.flatten()

            
            ## Fit Model & retrieve Feat Coefficients
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

            if self.model_name.lower() == 'xgb_classifier':
                model = XGBClassifier()
                y_scaled[abs(y_scaled) < 0.5] = 0
                y_scaled[abs(y_scaled) > 0.5] = 1
                model.fit(X_scaled, y_scaled)
                coef_df['Features'] = X_scaled.columns
                coef_df['Coeff'] = model.feature_importances_

            if self.model_name.lower() == 'xgb_regressor':
                model = XGBRegressor()
                model.fit(X_scaled, y_scaled)
                coef_df['Features'] = X_scaled.columns
                coef_df['Coeff'] = model.feature_importances_

            if self.model_name.lower() == 'pca':
                model = PCA()
                model.fit(X_scaled)
                # plt.plot(model.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
                # plt.title('Cumulative explained variance by number of principal components')
                # plt.savefig("./logs/pca.jpg")
                loadings = pd.DataFrame(data=model.components_.T * np.sqrt(model.explained_variance_),
                                        columns=[f'PC{i}' for i in range(1, len(X_scaled.columns) + 1)],
                                        index=X_scaled.columns)
                coef_df = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
                coef_df = coef_df.reset_index()
                coef_df.columns = ['Features', 'Coeff']

            logger.info("fit_predict executed with %s", self.model_name.lower())
        except Exception as e:
            logger.info("%s failed", self.model_name.lower())
            logger.error(e, exc_info=True)

        return coef_df

