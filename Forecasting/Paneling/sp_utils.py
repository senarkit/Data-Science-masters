"""
sp_utils:\n
This contains store paneling related utils or helper functions to perform aggregations and data formatting.
"""

import numpy as np
import pandas as pd
from utils import logger, names


def load_flat_file(filepath, format_cols=True, column_filter=None, encoding='utf-8',**kwargs):
    """

    To read the flat files

    Parameters
    ----------
    filepath : str
        input file path
    format_cols : bool, optional
        True for modifying the column names, by default True
    column_filter : list, optional
        List of column names to filter
    encoding : str, optional
        by default 'utf-8'

    Returns
    -------
    pd.DataFrame
        Data frame with formated column names
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath,encoding=encoding, **kwargs)
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, **kwargs)
        if filepath.endswith('.sas7bdat'):
            df = pd.read_sas(filepath,encoding=encoding,format='sas7bdat',**kwargs)
        
        if column_filter is not None:
            df = df[column_filter]

        if format_cols:
            df.columns = clean_col_names(df.columns,remove_chars_in_braces=False)
        return df
    except Exception as e:
        logger.info("%s - failed while reading", '')
        logger.error(e, exc_info=True)


def clean_col_names(string_series, special_chars_to_keep="_", remove_chars_in_braces=True, strip=True, lower=True):
    
    """
    Function to clean strings.

    Removes special characters, multiple spaces

    Parameters
    ----------
        string_series : pd.Series
        special_chars_to_keep : 
            string having special characters that have to be kept
        remove_chars_in_braces: 
            Logical if to keep strings in braces. e.g: "Packages (8oz)" will be "Packages"
        strip : True(default), 
            if False it will not remove extra/leading/tailing spaces
        lower : False(default), 
            if True it will convert all characters to lowercase

    Returns
    -------
        pandas series
    """
    try:
        if(lower):
            # Convert names to uppercase
            string_series = string_series.str.upper()
        if(remove_chars_in_braces):
            # Remove characters between square and round braces
            string_series = string_series.str.replace(r"\(.*\)|\[.*\]", '')
        else:
            # Add braces to special character list, so that they will not be
            # removed further
            special_chars_to_keep = special_chars_to_keep + "()[]"
        if(special_chars_to_keep):
            # Keep only alphanumeric character and some special
            # characters(.,_-&)
            reg_str = "[^\\w"+"\\".join(list(special_chars_to_keep))+" ]"
            string_series = string_series.str.replace(reg_str, '', regex=True)
        if (strip):
            # Remove multiple spaces
            string_series = string_series.str.replace(r'\s+', ' ', regex=True)
            # Remove leading and trailing spaces
            string_series = string_series.str.strip()
        string_series = string_series.str.replace(' ', '_')
        string_series = string_series.str.replace('_+', '_', regex=True)
        return(string_series)
    except AttributeError:
        print("Variable datatype is not string")
    except KeyError:
        print("Variable name mismatch")


def change_dtypes(df, type = 'string'):
    """
    To change the Numeric columns to numeric (converts non numeric entries to NaN) and non-Numeric columns to strings.

    Parameters
    ----------
    df : pd.DataFrame
        data frame in which the column dtypes are to be changed.
    type : str, optional
        'numeric' for Numeric columns and 'string' for others, by default 'string'

    Returns
    -------
    pd.DataFrame
        Returns same data frame after processing the dtypes
    """

    try:
        cols = df.columns
        if type == 'string':
            df[cols] = df[cols].astype(str)
            for col in cols:
                df[col] = np.where(df[col]=='nan', np.nan, df[col])
        if type == 'numeric':
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.info("%s - failed while reporting RF summary", '')
        logger.error(e, exc_info=True)


def weighted_avg_elasticity(hierarchy_data):
    """
    To calculate weighted average elasticity of the data frame that is passed. WAP = sum(pe*sales)/ sum(sales). This is put in groupby store x hierarchy to get hierarchywise weighted average elasticity for a store

    Parameters
    ----------
    hierarchy_data : pd.DataFrame
        data grouped by some category.

    Returns
    -------
    float
        Weighted average elasticity for the given grouped data
    """
    numerator = (hierarchy_data[names.unit_sales]*hierarchy_data[names.elasticity]).sum()
    denominator = hierarchy_data[names.unit_sales].sum()
    return numerator/denominator


def weighted_avg_price(hierarchy_data):
    """
    To calculate weighted average price of the data frame that is passed. WAP = Sales/ units. This is put in groupby weekend, hierarchy to get weeklevel hierarchywise weighted average price

    Parameters
    ----------
    hierarchy_data : pd.DataFrame
        data grouped by some category.

    Returns
    -------
    float
        Weighted average Price for the given grouped data
    """
    # numerator = (hierarchy_data['ALACARTE_UNITS']*hierarchy_data[price_col]).sum()
    numerator = (hierarchy_data[names.sales]).sum()
    denominator = hierarchy_data[names.unit_sales].sum()
    return numerator/denominator
