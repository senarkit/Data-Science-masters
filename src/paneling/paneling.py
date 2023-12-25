"""
Panelling:\n
Panelling.py contains all the important functions used in store_paneling module. \n
It contains functions to get data from Feature selection module, run elasticity modules (if required), run GMM or RF modules and generate reports for validations.
"""

import os
import traceback
import warnings

import numpy as np
import pandas as pd
from features.fs import fs_paneling
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import config, logger, names
from utils.file_utils import init

import store_paneling.sp_utils as ut
# from store_paneling.FE import get_data, get_store_additional_features
from store_paneling.models import Lme, gmm, rf

warnings.filterwarnings("ignore")



class clustering:
    """
    Class that contains most of the self variables and functions related to store_paneling. Following is the description of these variables:\n
    1. self.fe_for_paneling: It contains data for all the features to be fed to GMM or RF models. Generated from either process_data() or get_custom_input(). It is used in elbow(), get_cluster(), gmm_reporting(), classifier() and rf_reporting().\n
    2. self.fe_for_elasticity: It contains data for all the features to be fed to LME. Applicable only when use_demand_elasticity or use_gc_elasticity is True. Used only in get_elasticities() function.\n
    3. self.folder_dict: It contains paths of all the folders related to store_paneling. used in all reporting related functions.
    """

    def __init__(self):
        """
        To initialize the clustering class. It generates paths the output folder structure and generates some basic self variables like local to global id mapping, and item to hierarchy mapping.
        """
        try:
            self.folder_dict = init(by='store_paneling')
            storeinfo = ut.load_flat_file(config.path.store_info)
            self.global_to_local_dict = dict(storeinfo[[names.global_id, names.store_id]].values)
            self.local_to_global_dict = dict(storeinfo[[names.store_id, names.global_id]].values)
            
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed during initializing clustering class", '')
            logger.error(e, exc_info=True)
            

    def process_data(self):
        """process_data
        
        Process_data collates all the features from PMIX, GC, elasticities, store info etc. to generate required features
        

        Returns
        -------
        None
            Generates a self variable (self.fe_for_paneling) that contains store level data for the final features
        """
        try:
            print('##### Processing Data #####')

            # Get all PMIX and GC related features from FS and apply date filters
            start_date = pd.to_datetime(config.paneling.run.start_date)
            end_date = pd.to_datetime(config.paneling.run.end_date)
            
            self.pmix, self.gc, self.fe_for_paneling, stores_final = fs_paneling(start_date, end_date)
            
            # To run LME models for store level elasticities
            if bool(config.fs.paneling.use_demand_elasticity) or bool(config.fs.paneling.use_gc_elasticity):
                self.fe_for_elasticity = self.gc.copy()
                self.fe_for_elasticity = self.fe_for_elasticity[self.fe_for_elasticity[names.global_id].isin(stores_final)]
                self.fe_for_elasticity[names.week_id] = pd.to_datetime(self.fe_for_elasticity[names.week_id])               
                
                units_df = self.pmix.groupby([names.global_id, names.week_id])[names.unit_sales].sum().reset_index()
                self.fe_for_elasticity = self.fe_for_elasticity.merge(units_df, on=[names.global_id, names.week_id], how='left')

                if bool(config.fs.paneling.use_demand_elasticity):
                    demand_elasticity_dict = self.get_elasticities(y_var='units')
                    self.fe_for_paneling['DEMAND_ELASTICITY'] = self.fe_for_paneling[names.global_id].map(demand_elasticity_dict)
                    if config.fs.paneling.wap=='bins':
                        self.fe_for_paneling['DEMAND_ELASTICITY'] = pd.qcut(self.fe_for_paneling['DEMAND_ELASTICITY'],4)

                if bool(config.fs.paneling.use_gc_elasticity):
                    gc_elasticity_dict = self.get_elasticities(y_var='gc')
                    self.fe_for_paneling['GC_ELASTICITY'] = self.fe_for_paneling[names.global_id].map(gc_elasticity_dict)
                    if config.fs.paneling.wap=='bins':
                        self.fe_for_paneling['GC_ELASTICITY'] = pd.qcut(self.fe_for_paneling['GC_ELASTICITY'],4)

            self.fe_for_paneling = self.fe_for_paneling.fillna(self.fe_for_paneling.median())
            self.fe_for_paneling = self.fe_for_paneling.set_index([names.global_id, names.store_id])

            cols_to_drop = self.fe_for_paneling.columns[self.fe_for_paneling.nunique()==1]
            self.fe_for_paneling.drop(columns=cols_to_drop, inplace=True)

            print("Created Features for GMM")
            
            return None
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed during Processing data", '')
            logger.error(e, exc_info=True)


    def get_elasticities(self, y_var='units'):
        """get_elasticities

        To get the demand or gc elasticities for all the stores. It uses store x week level data to predict units or gc and returns of store-elasticity mapping for each store.

        Parameters
        ----------
        y_var : str, optional
            "units" for demand elasticity and "gc" for GC elasticity, by default 'units'

        Returns
        -------
        Dict
            Store level wap coefficients

        """
        try:
            print('##### Getting store elasticities: ' + y_var + ' #####')
            # log transform price var
            fe_for_elasticity = self.fe_for_elasticity.copy()
            fe_for_elasticity['WAP'] = np.log(fe_for_elasticity['WAP'])
            fe_for_elasticity = fe_for_elasticity.sort_values(
                [names.global_id, names.week_id]).reset_index(drop=True)

            fe_for_elasticity['index_'] = fe_for_elasticity[names.global_id].astype(str) + "_" + fe_for_elasticity[names.week_id].astype(str)
            fe_for_elasticity = fe_for_elasticity.set_index('index_')
            fe_for_elasticity.drop(columns=names.week_id, inplace=True)
            fe_for_elasticity.dropna(inplace=True)


            lme_args = {}
            # log transform y_var and drop other y column
            if y_var == 'units':
                fe_for_elasticity[names.unit_sales] = np.log(fe_for_elasticity[names.unit_sales]/fe_for_elasticity[names.days_open])
                fe_for_elasticity.drop(columns=[names.gc, names.days_open, names.store_id], inplace=True)
                lme_args['y_col'] = names.unit_sales

            if y_var == 'gc':
                fe_for_elasticity[names.gc] = np.log(fe_for_elasticity[names.gc]/fe_for_elasticity[names.days_open])
                fe_for_elasticity.drop(columns=[names.unit_sales, names.days_open, names.store_id], inplace=True)
                lme_args['y_col'] = names.gc

            # Check for correlation
            corr_matrix = fe_for_elasticity.drop(columns=[names.global_id, 'WAP', lme_args['y_col']]).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
            fe_for_elasticity.drop(to_drop, axis=1, inplace=True)
            fe_for_elasticity.drop(list(fe_for_elasticity.columns[fe_for_elasticity.nunique()==1]), axis=1, inplace=True)

            # Check for singularity
            cols_to_drop = fe_for_elasticity.columns[fe_for_elasticity.nunique()==1]
            fe_for_elasticity.drop(columns=cols_to_drop, inplace=True)

            train = fe_for_elasticity.copy()


            lme_args['group_col'] = names.global_id
            lme_args['random_effects'] = [names.__dict__[col] for col in config.elasticity_est.lme_config.random_effects]
            lme_args['fixed_effects'] = [col for col in fe_for_elasticity.columns if col not in [lme_args['group_col']]+[lme_args['y_col']]+lme_args['random_effects']]

            # Scaling the features
            # scaler = MinMaxScaler()
            # scaled_train = scaler.fit_transform(train)
            # scaled_test = scaler.transform(test)

            # y_col           : column name of dependent variable (str)
            # group_col       : column name containing grouping variable for random fixed_effects (str)
            # fixed_effects   : list of features to be treated as fixed effects
            # random_effects  : list of features to be treated as random effects

            LME = Lme(train, lme_args, track_tag='')
            res = LME.fit()
            self.params = LME.get_params()
            self.pred_df = LME.predict(train)
            self.pred_df['index_'] = train.index
            self.pred_df[names.week_id] = self.pred_df['index_'].apply(lambda x: pd.to_datetime(x.split("_")[1]))

            re_coeff = self.params[1]
            elasticity_dict = dict(zip(re_coeff[names.global_id],re_coeff['WAP']))

            train['MODEL_FLAG'] = 'TRAIN'
            self.data_used_for_lme = train.copy()
            
            self.lme_reporting(y_var)
            return elasticity_dict
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed during Elasticity calculations", '')
            logger.error(e, exc_info=True)


    def get_custom_input(self):
        """get_custom_input
        
        Inputs the Features from a flat file instead of generating them from input data

        Returns
        -------
        None
            creates a self variable (self.fe_for_paneling) that contains store level data for the final features
        """
        try:
            self.fe_for_paneling = ut.load_flat_file(config.path.custom_input_for_paneling_features_path)
            
            if names.global_id not in self.fe_for_paneling.columns:
                self.fe_for_paneling[names.global_id] = self.fe_for_paneling[names.store_id].map(self.local_to_global_dict)
            if names.store_id not in self.fe_for_paneling.columns:
                self.fe_for_paneling[names.store_id] = self.fe_for_paneling[names.global_id].map(self.global_to_local_dict)

            self.fe_for_paneling = self.fe_for_paneling.fillna(self.fe_for_paneling.median())
            self.fe_for_paneling = self.fe_for_paneling.set_index([names.global_id, names.store_id])            
            return None
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while importing custom input", '')
            logger.error(e, exc_info=True)


    def elbow(self, max_clusters):
        """elbow

        Generates an elbow curve and saves it as a pdf in output folder

        Parameters
        ----------
        max_clusters : int
            maximum number of clusters in then elbow curve.

        Returns
        -------
        None
            Saves a file and image of the elbow curve to choose the optimal number of clusters.
            
        """
        try:
            X = pd.get_dummies(self.fe_for_paneling.dropna(axis=1), drop_first=True)
            wcss = []
            for i in range(1, max_clusters):
                kmean = KMeans(n_clusters=i, init="k-means++")
                kmean.fit_predict(X)
                wcss.append(kmean.inertia_)

            plt.plot(range(1, max_clusters), wcss)
            plt.title('Elbow Curve')
            plt.xlabel("No of Clusters")
            plt.ylabel("WCSS")
            plt.savefig(os.path.join(self.folder_dict['Output_GMM'], 'elbow.pdf'), bbox_inches='tight')
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed during elbow curve generation", '')
            logger.error(e, exc_info=True)


    def get_clusters(self):
        """get_clusters

        Scale and Run the GMM model on store level data to get the store-panel mapping and summary about the model Parameters

        Returns
        -------
        dict
            {num_cluster:{cluster_dict['Summary']:gmm_summary, cluster_dict['Labels']:labels}}
        """
        try:
            print('##### Getting Clusters from GMM #####')
            num_clusters = [2, 3, 4]  # Get it from elbow function

            data = pd.get_dummies(self.fe_for_paneling.dropna(axis=1), drop_first=True)
            # data.to_excel(os.path.join(self.folder_dict['Model_data_GMM'], 'before_scaling.xlsx'), index=False)

            # Scale featurs to convert to normal dist.
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(data)
            self.scaled_data = pd.DataFrame(self.scaled_data, index=data.index, columns=data.columns)
            # self.scaled_data = data.copy()
            self.final_dict_gmm = {}
            for clusters in num_clusters:
                gmm_args = {}
                gmm_args['num_clusters'] = clusters
                gmm_args['covariance_type'] = config.model.gmm.covariance_type
                # gmm_args['covariance_type'] = 'spherical'
                gmm_args['max_iter'] = config.model.gmm.max_iter
                gmm_args['init_params'] = config.model.gmm.init_params

                GMM = gmm(self.scaled_data, gmm_args)
                GMM.fit()
                labels = GMM.predict()
                gmm_summary = GMM.get_model_summary()

                cluster_dict = {}
                cluster_dict['Summary'] = gmm_summary
                cluster_dict['Labels'] = labels

                self.final_dict_gmm[clusters] = cluster_dict

            # for clusters in num_clusters:
            #     print(self.final_dict_gmm[clusters]['Summary'])
            return self.final_dict_gmm
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while running GMM", '')
            logger.error(e, exc_info=True)


    def classifier(self):
        """
        Runs RF classifier models on the given data to get store-panel mapping for new stores and summary about the model Parameters

        Returns
        -------
        Dictionary
            {num_cluster:{cluster_dict['Summary']:rf_summary, cluster_dict['Labels']:labels}}
        """
        
        try:
            print('##### Running RF model #####')

            rf_args = {}
            rf_args['y_col'] = names.panel
            rf_args['n_estimators'] = [int(x) for x in np.linspace(start=config.model.rf.n_estimators.start, stop=config.model.rf.n_estimators.stop, num=config.model.rf.n_estimators.num_of_samples)]
            rf_args['max_features'] = [config.model.rf.max_features]
            rf_args['max_depth'] = [int(x) for x in np.linspace(start=config.model.rf.max_depth.start, stop=config.model.rf.max_depth.stop, num=config.model.rf.max_depth.num_of_samples)]+[None]
            rf_args['min_samples_split'] = np.arange(config.model.rf.min_samples_split.start, config.model.rf.min_samples_split.stop)
            rf_args['min_samples_leaf'] = np.arange(config.model.rf.min_samples_leaf.start, config.model.rf.min_samples_leaf.stop)
            rf_args['bootstrap'] = config.model.rf.bootstrap


            current_mapping = ut.load_flat_file(config.path.existing_store_panel_mapping)
            mapping = dict(current_mapping[[names.global_id,names.panel]].values)
            

            fe_for_rf_dummies = pd.get_dummies(self.fe_for_paneling, drop_first=True).reset_index()
            fe_for_rf_dummies[names.panel] = fe_for_rf_dummies[names.global_id].map(mapping)
            fe_for_rf_dummies.set_index([names.global_id,names.store_id], inplace=True)
            
            self.fe_for_rf = fe_for_rf_dummies
            train_rf = fe_for_rf_dummies[~fe_for_rf_dummies[names.panel].isna()]
            test_rf = fe_for_rf_dummies[fe_for_rf_dummies[names.panel].isna()].drop(columns=names.panel)

            self.teststores = test_rf.reset_index()[names.global_id].values
            RF = rf(train_rf, test_rf, rf_args)
            rf_mapping = RF.predict()
            rf_summary = RF.get_model_summary()

            rf_dict = {}
            rf_dict['Labels'] = rf_mapping
            rf_dict['Summary'] = rf_summary

            self.final_dict_rf = rf_dict

            return self.final_dict_rf
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while running RF", '')
            logger.error(e, exc_info=True)
        

    def lme_reporting(self, y_var):
        """
        To save all the features actuals and predictions from the LME models for validation.

        Returns
        -------
        None
            Saves an excel file to summerize the LME models in the output folder
        """
        
        try:
            if config.paneling.run.run_gmm:
                writer = pd.ExcelWriter(os.path.join(self.folder_dict['Output_GMM'],y_var+ '_LME_summary.xlsx'), engine='xlsxwriter')
            if config.paneling.run.run_rf:
                writer = pd.ExcelWriter(os.path.join(self.folder_dict['Output_RF'],y_var+ '_LME_summary.xlsx'), engine='xlsxwriter')
            self.params[1].to_excel(writer, sheet_name='Elasticities', index=False)
            self.params[0].to_excel(writer, sheet_name='Features', index=False)
            self.pred_df.to_excel(writer, sheet_name='Predictions', index=False)
            self.data_used_for_lme.to_excel(writer, sheet_name='Data used for LME')
            writer.save()
            return None
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while reporting LME summary", '')
            logger.error(e, exc_info=True)


    def gmm_reporting(self):
        """
        To generate the GMM summary excels, pivots for further segregation of clusters and save in outpath. The output files contain Final store-panel mapping, model related details and store distribution in each cluster x feature.
        """
        try:
            data = pd.get_dummies(self.fe_for_paneling.dropna(axis=1), drop_first=True)
            data.to_excel(os.path.join(self.folder_dict['Model_data_GMM'], 'raw_input.xlsx'))
            self.scaled_data.to_excel(os.path.join(self.folder_dict['Model_data_GMM'], 'scaled_input.xlsx'))

            for num_cluster in list(self.final_dict_gmm.keys()):
                writer = pd.ExcelWriter(os.path.join(self.folder_dict['Output_GMM'], 'gmm_out_' + str(num_cluster)+'_clusters.xlsx'), engine='xlsxwriter')
                cluster_dict = self.final_dict_gmm[num_cluster]
                gmm_summary = cluster_dict['Summary']

                # weights = pd.DataFrame(gmm_summary['Weights'], columns=['Weights'])
                # weights.to_excel(writer, sheet_name='Weights')

                # means = pd.DataFrame(gmm_summary['Means'])
                # means.to_excel(writer, sheet_name='Means')

                # other_summary = pd.DataFrame([gmm_summary['Converged'], gmm_summary['No. of Iterations'], gmm_summary['Lower Bound']], index=['Converged', 'No. of Iterations', 'Lower Bound'])
                other_summary = pd.DataFrame([len(cluster_dict['Labels']),gmm_summary['Converged'], gmm_summary['No. of Iterations']], index=['No. of stores modelled','Converged', 'No. of Iterations'])
                other_summary.to_excel(writer, sheet_name='Summary')

                labels = cluster_dict['Labels']
                mapping = pd.DataFrame(labels, index=self.fe_for_paneling.index,columns=[names.panel]).reset_index()
                
                # mapping.columns = [names.global_id, names.panel]
                mapping.to_excel(writer, sheet_name='Mapping', index=False)

                store_counts = mapping[names.panel].value_counts().reset_index()
                store_counts.columns = [[names.panel, 'STORE COUNTS']]
                store_counts.to_excel(writer, sheet_name='Store counts', index=False)

                pivot_data = self.fe_for_paneling.copy()
                pivot_data[names.panel] = labels
                
                pivot_data_num = pivot_data.select_dtypes(include=np.number)
                num_cols = pivot_data_num.columns
                pivot_data_num = pivot_data_num.reset_index()
                
                startrow=0
                for col in num_cols:
                    if col != names.panel:
                        pivot_data_num[col+'_BINS'] = pd.qcut(pivot_data_num[col], 4, duplicates='drop')
                        df = pivot_data_num.pivot_table(index=col+'_BINS', columns=names.panel, values=names.global_id,
                                                aggfunc=pd.Series.nunique).reset_index()
                        df.to_excel(writer, sheet_name="Num Col Pivot", index=False, startrow=startrow)
                        startrow = startrow + 3 + df.shape[0]

                cat_cols = [col for col in pivot_data.columns if col not in num_cols]
                pivot_data = pivot_data.reset_index()
                pivot_data_cat = pivot_data[[names.global_id, names.panel]+cat_cols]
                startrow=0
                for col in cat_cols:
                    df = pivot_data_cat.pivot_table(index=col, columns=names.panel, values=names.global_id,
                                            aggfunc=pd.Series.nunique).reset_index()
                    df.to_excel(writer, sheet_name="Cat Col Pivot", index=False, startrow=startrow)
                    startrow = startrow + 3 + df.shape[0]


                # panel_counts= pivot_data.groupby(names.panel)[names.global_id].nunique().reset_index()
                # panel_counts.to_excel(writer, sheet_name="Panel counts", index=False)

                writer.save()
        except Exception as e:
            # print(traceback.format_exc())
            logger.info("%s - failed while reporting GMM summary", '')
            logger.error(e, exc_info=True)


    def rf_reporting(self):
        """
        To generate the RF summary excels, pivots for further segregation of clusters and save in outpath. The output files contain Final store-panel mapping, model related details and store distribution in each cluster x feature.

        """
        try:
            self.fe_for_rf.to_excel(os.path.join(self.folder_dict['Model_data_RF'], 'raw_input_rf.xlsx'))
            
            writer = pd.ExcelWriter(os.path.join(self.folder_dict['Output_RF'], 'RF_Summary.xlsx'), engine='xlsxwriter')
            rf_dict = self.final_dict_rf
            rf_summary = rf_dict['Summary']
            labels = rf_dict['Labels']

            pd.DataFrame(rf_summary['Best_Params'], index=[0]).T.to_excel(writer, sheet_name='Best_Params')
            rf_summary['Feature_importance'].to_excel(writer, sheet_name='Feature_importance')

            labels['MODEL_FLAG'] = np.where(labels[names.global_id].isin(self.teststores), 'NEW', 'OLD')
            labels.to_excel(writer, sheet_name='mapping')

            pivot_data = self.fe_for_rf.reset_index().copy()
            teststores = labels[labels['OLD_CLUSTER'].isna()][names.global_id].to_list()
            mapping = dict(labels[[names.global_id, 'FINAL_CLUSTER']].values)
            pivot_data[names.panel] = pivot_data[names.global_id].map(mapping)

            pivot_data_num = pivot_data.select_dtypes(include=np.number)
            num_cols = pivot_data_num.columns
            pivot_data_num = pivot_data_num.reset_index()
            pivot_data_num['MODEL_FLAG'] = np.where(pivot_data_num[names.global_id].isin(teststores), 'NEW', 'OLD')
            
            startrow=0
            for col in num_cols:
                if col != names.panel:
                    pivot_data_num[col+'_BINS'] = pd.qcut(pivot_data_num[col], 4, duplicates='drop')
                    df = pivot_data_num.pivot_table(index=[col+'_BINS', 'MODEL_FLAG'], columns=names.panel, values=names.global_id,
                                            aggfunc=pd.Series.nunique).reset_index()
                    df.to_excel(writer, sheet_name="Num Col Pivot", index=False, startrow=startrow)
                    startrow = startrow + 3 + df.shape[0]

            cat_cols = [col for col in pivot_data.columns if col not in num_cols]
            pivot_data = pivot_data.reset_index()
            pivot_data['MODEL_FLAG'] = np.where(pivot_data[names.global_id].isin(teststores), 'NEW', 'OLD')
            pivot_data_cat = pivot_data[[names.global_id, names.panel, 'MODEL_FLAG']+cat_cols]
            startrow=0
            for col in cat_cols:
                df = pivot_data_cat.pivot_table(index=[col, 'MODEL_FLAG'], columns=names.panel, values=names.global_id,
                                        aggfunc=pd.Series.nunique).reset_index()
                df.to_excel(writer, sheet_name="Cat Col Pivot", index=False, startrow=startrow)
                startrow = startrow + 3 + df.shape[0]


            pivot_data.groupby(['MODEL_FLAG',names.panel])[names.global_id].nunique().reset_index().to_excel(writer, sheet_name="Cluster counts")

            writer.save()
        except Exception as e:
            # print(traceback.format_exc())
            writer.save()
            logger.info("%s - failed while reporting RF summary", '')
            logger.error(e, exc_info=True)
