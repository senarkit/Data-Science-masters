"""
sp_driver:\n
sp_driver.py is the enrty point for store paneling module. It calls appropriate fuction about whether to run GMM or RF based on the config entries.
"""
import traceback

from utils import config, logger

from store_paneling.panelling import clustering


def run_store_paneling():
    """
    Calls the necessary functions to run GMM or RF. It also decides how model data for paneling is generated based on the config entries.
    """
    try:
        obj = clustering()

        if config.paneling.run.custom_input_for_features:
            obj.get_custom_input()
        else:
            obj.process_data()
        if config.paneling.run.run_gmm:
            obj.elbow(6)
            obj.get_clusters()
            obj.gmm_reporting()
        if config.paneling.run.run_rf:
            obj.classifier()
            obj.rf_reporting()
    except Exception as e:
        # print(traceback.format_exc())
        logger.info("%s - failed while running store paneling", '')
        logger.error(e, exc_info=True)
