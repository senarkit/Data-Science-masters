

import logging
import datetime


### LOGGING
now = datetime.datetime.now().strftime("%d-%m-%Y-%HH:%MM:%SS")
logging.basicConfig(filename=f"./logs/LOG_{now}.log", 
                    filemode='w', 
                    level = logging.DEBUG,
                    format='[%(asctime)s] : %(name)s - %(funcName)s | %(levelname)s | %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.debug('START main() script execution')

## main script code/function calls


## END
logger.debug("END main() script execution")