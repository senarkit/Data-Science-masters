import logging
import datetime
import yaml
import pandas as pd
from Features.selection import feature_selection

#### import config file
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

#### LOGGING
now = datetime.datetime.now().strftime("%d-%m-%Y-%HH:%MM:%SS")
logging.basicConfig(filename=f"./logs/LOG_{now}.log", 
                    filemode='w', 
                    level = logging.DEBUG,
                    format='[%(asctime)s] : %(name)s - %(funcName)s | %(levelname)s | %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.debug('START main() script execution')


#### main script code/function calls
data = pd.read_csv(config['path']['filepath'])
y = data[config['fs']['target']].copy()
X = data[config['fs']['features']]

## Feature Selection
fs = feature_selection(X, y, config['fs']['features'], config['fs']['model'])
feat_coeff = fs.fit_predict()
feat_coeff['coeff_abs'] = abs(feat_coeff['Coeff'])
feat_coeff = feat_coeff.sort_values(by=['coeff_abs'], ascending=[False])
print(feat_coeff)

#### END
logger.debug("END main() script execution")