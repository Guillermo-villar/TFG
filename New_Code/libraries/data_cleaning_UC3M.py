# -*- coding: utf-8 -*-

# copyright GMV 2022
# created by ggnn

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries
import pandas as pd
import numpy as np
import re
import sys
import logging
import string

import pdb

# add project path
# sys.path.append('../')

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# ------------------------------------------------------------------------------------------------------------------
# DATA COMPRESSION
# ------------------------------------------------------------------------------------------------------------------
def get_seq_events(df, timestamp_colname=None, col_to_group=None, 
                   start_time=None, end_time=None, 
                   col_to_fuse=None,
                   max_event_sep=5*60):
    
    if timestamp_colname == None:
        timestamp_colname = 'TimeOfArrival'
        
    if col_to_group == None:
        col_to_group = ['ElemServID']    
    
    if col_to_fuse == None:
        fuse_flag = False
    else:
        fuse_flag = True
        
    if start_time == None:
        start_time = df.iloc[0][timestamp_colname]
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time, format='%Y-%m-%d')
        
    if end_time == None:
        end_time = df.iloc[-1][timestamp_colname]
    elif isinstance(end_time, str):
        end_time = pd.to_datetime(end_time, format='%Y-%m-%d')
    
    DeltaT = pd.Timedelta(seconds=max_event_sep)
    
    df_sorted = df.sort_values(by=[timestamp_colname]).reset_index(drop=True)
    
    df_o = pd.DataFrame()
    flag_list = False
    if isinstance(col_to_group, list):
        df_c = df_sorted[col_to_group].apply(tuple, axis=1)
        df_c_unique = df_c.unique()
        flag_list = True
    elif isinstance(col_to_group, str):
        df_c_unique = df_sorted[col_to_group].unique()
    
    for val in df_c_unique:
        if flag_list:
            df_s = df_sorted[df_c == val]
        else:
            df_s = df[df[col_to_group]==val]
        id_s = np.where(1-(pd.to_datetime(df_s[timestamp_colname]).diff() < DeltaT))[0]
        length_s = np.append(np.diff(id_s),df_s.shape[0]-id_s[-1])
        df_ok = df_s.iloc[id_s].reset_index(drop=True).copy()
        # if 'EndSeq' in df_ok.columns:
        #     df_ok = df_ok.rename(columns={'EndSeq': 'EndAlarm', 'LenSeq': 'LenAlarm'})
        df_ok.insert(loc=2,column='EndEvent', value = df_s.iloc[id_s+length_s-1][timestamp_colname].reset_index(drop=True))
        df_ok.insert(loc=3,column='LenEvent', value = length_s)
        
        if fuse_flag:
            col_of_interest = col_to_fuse[0]
            if col_to_fuse[1]=='int':
                data_type = int
            elif col_to_fuse[1]=='float':
                data_type = float
            elif col_to_fuse[1]=='str':
                data_type = str
                for k_idx in range(len(id_s)-1):
                    new_text = list(set("".join(df_s.iloc[id_s[k_idx]:id_s[k_idx+1]][col_of_interest].to_list()).split(' ')))
                    df_ok.loc[k_idx, col_of_interest] = " ".join(new_text)
            else:
                data_type = None
            if (data_type == int) | (data_type==float):
                data_op = col_to_fuse[2]
                for k_idx in range(len(id_s)-1):
                    values_to_fuse = df_s.iloc[id_s[k_idx]:id_s[k_idx+1]][col_of_interest].to_list()
                    output = data_type(eval(data_op)(values_to_fuse))
                    df_ok.loc[k_idx, col_of_interest] = output

        df_o = pd.concat([df_o, df_ok], axis=0)
            
    return df_o

# ------------------------------------------------------------------------------------------------------------------
# FILTER OUT NANS
# ------------------------------------------------------------------------------------------------------------------
def count_and_filter_nans(df ,thr_nans_ratio=1):
    """Given a daframe DF, check the % of NaN in all features
    Moreover, if defining THR_NANS_RATIO lower than 1 (0-1) it allows for
    filtering those features that result too sparse

    Args:
            df             (DataFrame) pandas dataframe
            thr_nans_ratio (float)     threshold for filtering variables affected by nans

    Returns:
            nans_df        (DataFrame) a one column dataframe containing % of nans in the original dataframe
            filtered_df    (DataFrame) filtered dataframe after removal of sparse features

    """

    n_df = len(df)
    nan  = df.isnull().sum()
    nan_perc= nan * 100 / n_df
    # for each file paint a bar chart of features for NaNs % (sorted)
    nans_df = pd.DataFrame(nan_perc,
                           index=df.columns, columns=['nan_perc'])
    filtered_df = df[[feat for feat in list(nans_df[nans_df['nan_perc' ] <= 100 * thr_nans_ratio].index)]]

    return nans_df, filtered_df


# ------------------------------------------------------------------------------------------------------------------
# COUNT UNIQUE VALUES
# ------------------------------------------------------------------------------------------------------------------
def count_unique_numb(df, thr_nunique_ratio=1, keep_constant_flag=False):
    """ Count the number of unique values for each numeric feature.
    Filter those feature with a threshold of unique ratio < thr_nunique_ratio
    The filtered dataset is returned as well.

    Args:
            df (dataFrame)              dataframe
            thr_nunique_ratio (float)   floating value for defining the (at least one)
                                        not unique ratio caused by vert git temperature
            keep_constant_flag (bool)   flag to avoid

    Returns:
            nunique_df (dataFrame) number of unique data columns

        """

    # pandas describe
    n_df = len(df)

    # number of unique values (for identifying categorical values)
    nunique_df = pd.DataFrame(df.nunique().values,
                              index=df.nunique().index, columns=['num_unique_values'])
    nunique_df.dropna(axis=0, inplace=True)

    filtered_df = df[[feat for feat in list(nunique_df.index) if (
                (nunique_df.loc[feat].values <= n_df * thr_nunique_ratio)  # --> features without full variability <---
                and (keep_constant_flag or (nunique_df.loc[feat].values > 1)))]]

    return nunique_df, filtered_df


# --------------------------------------------------------------------------------------------------
# REGEX AND TEXT PARSING FUNCTIONS
# --------------------------------------------------------------------------------------------------

def binarize(x, w=8, missing_name='missing_ip_8bits'):
    """ Binarize the IP number in X using words of W number of bits

    Args:
        x (str) input string
        w (int) number of bits for the input string
        missing_name (str) string for the missing fiename saving

    Returns:
        binarized representation of the input or missing_name value

    """
    # x = re.search('[^.]*([0-9]*[[0-9]+]?)[^.]', str(x))
    x = re.search(r'\d+',str(x))

    if not x:

        return missing_name

    else:

        return np.binary_repr(int(x[0]), width=w)

# --------------------------------------------------------------------------------------------------


def extract_numeric(x, missing_name='missing_numeric'):
    """ Extract numeric feature value as X from a real value dataframe (series)
    and assign a MISSING_NUMERIC value to thata feature

    Args:
        x            (str) input string
        missing_name (str) missing value identifier

    Returns:
        float value or missing value identifier

    """

    x = re.search('[0-9]*\.*[0-9]+', str(x))

    if not x:
        return missing_name
    else:
        return float(x[0])

# --------------------------------------------------------------------------------------------------


def extract_alphanum_template(x, missing_name='missing_alphanum'):
    """ Extract the alfanumeric value from a digit input X and and also
    assign MISSING_ALPHANUM value for those features being alfanumeric

    Args:
            x            (str) input string
            missing_name (str) missing value identifier

    Returns:

            parsed alfanumeric value of missing value identifier

    """
    try:
        return re.sub(r"\d+\.*\d*", "*", str(x))

    except Exception as e:

        return missing_name

# --------------------------------------------------------------------------------------------------


def split_composed_n_parts(x, n, split_char='-', missing_name='missing_composed'):
    """ Split a composed input string X and return a list of components
     and MISSING_COMPOSED string for those values that were not assigned """    
    if isinstance(x,str):
        x = x.lower().replace('ñ','n').split(split_char)
        w = len(x)
    else: # this is the case of missing values at all
        w = 0
        
    z = []

    for ii in range(n):
        if ii>w-1:
            z.append(missing_name+'_'+str(ii+1))
        elif x[ii]==np.nan and x[ii]=='None':
            z.append(missing_name+'_'+str(ii+1))
        else:
            z.append(x[ii]) # .split(' - ')[0]) # site (subsytem part: e.g. Torrespaña)
            # z.append(x[ii].split('-')[1]) # neglect the site part
            
    z.append(w)
            
    return tuple(z)

# --------------------------------------------------------------------------------------------------


def param_key_cleaning(x, n, missing_name='missing_param_key'):
    """ Clean a parameter key by adding MISSING_NAME string to those values
    that are not explicitely attributed (assigned)"""

    if isinstance(x, str):
        x = x.lower().replace('ñ', 'n').replace('.', '_').replace(' ', '_').split('/')
        w = len(x)
    else:  # this is the case of missing values at all
        w = 0

    z = []
    
    for ii in range(n):
        if ii > w - 1:
            z.append(missing_name + '_' + str(ii + 1))
        elif x[ii] == np.nan and x[ii] == 'None':
            z.append(missing_name + '_' + str(ii + 1))
        else:
            first = x[ii].split('-')[0]  # first element is key
            x = re.search('[0-9]*[\.|\,]*[0-9]+', str(first))
            if not x:
                z.append(first.rstrip('[0-9]*'))
            else:
                x = re.search("^[\d]*$", str(x[0]))
                if not x:
                    z.append('float_value')
                else:
                    z.append('int_value')

    return z[0]


# --------------------------------------------------------------------------------------------------

def param_key_cleaning_UC3M(x, n, site_cod_dict, serv_equiv_dict, missing_name='missing_param_key'):
    """ Clean a parameter key by adding MISSING_NAME string to those values
    that are not explicitely attributed (assigned)"""

    sites_list = [site_cod_dict[x]['SiteCodInfo'] for x in range(len(site_cod_dict))]
    if isinstance(x, str):
        x = x.lower().replace('ñ', 'n') #.replace('.', '_').replace(' ', '_').split('/')
        x_slash = x.split('/')
        x_dot = x.split('.')
        x_dash = x.split(' - ')
        lx_slash = len(x_slash)
        lx_dot = len(x_dot)
        lx_dash = len(x_dash)
        
        # Detect Pattern
        
        if lx_slash >=3:
            pat_x = 0
            pat_name  = '12600/00000/AAA'
        else:
            if lx_dash == 3:
                pat_x = 1
                pat_name  = 'RF - CH - Service'
            else:
                if lx_dot==3:
                    pat_name  = 'xxxx.y.z'
                    pat_x = 2
                elif lx_dot == 2:
                    pat_name  = 'xxx.x'
                    pat_x = 2
                else:    
                    if re.compile(r'^\d+(?:,\d*)?$').match(x_dot[0]):
                        pat_name  = 'xxxx'
                        pat_x = 4
                    else:
                        pat_name = 'unknown 5'
                        pat_x = 5

    else:  # this is the case of missing values at all
        z = None
        return z

    z = []
    z_serv_ini = None
    z_params = None

    if pat_x == 0: # pat_name  = '12600/00000/AAA'
        elem_cod_list = []
        x = x.replace('.', '_').replace(' ', '_').split('/')
        lx = len(x)
        key_found = False
        site_found = False
        elem_found = False
        energy_found = False
        monitor_found = False
        transmitter_found = False
        if lx > 0:
            ii = 0
            if x[ii] == np.nan or x[ii] == 'None' or x[ii] == '':
                z.append(missing_name + '_' + str(ii + 1))
            else:
                first = x[ii].split('-')[0]  # first element is key
                xf = re.search('[0-9]*[\.|\,]*[0-9]+', str(first))
                if not xf:
                    z = x[ii:]
                    # z.append(zkey)
                    key_found = True
                else:
                    if xf[0] in sites_list: # site_cod_dict.keys():
                        site_cod = xf[0]
                        site_found = True
            if lx > 1:
                ii = 1
                if x[ii] == np.nan or x[ii] == 'None' or x[ii] == '':
                    z.append(missing_name + '_' + str(ii + 1))
                else:
                    first = x[ii].split('-')[0]  # first element is key
                    xf = re.search('[0-9]*[\.|\,]*[0-9]+', str(first))
                    if not xf:
                        z = x[ii:]
                        # z.append(zkey)
                        key_found = True
                        # Check if item is related to climate, energy, access
                        if x[ii] in ['climate', 'access', 'energy']:
                            energy_found = True
                        elif x[ii] in ['dtt_monitor']:
                            monitor_found = True
                        elif x[ii] in ['transmitter']:
                            monitor_found = True
                    # else:
                    #     elem_cod = xf[0]
                    #     elem_cod_list.append(elem_cod)
                    #     elem_found = True
            else:
                # print(1)
                z.append(missing_name + '_' + str(ii + 1))

        ii = 2                            
        while not(key_found) and ii < lx:
            if x[ii] == np.nan or x[ii] == 'None' or x[ii] == '':
                z.append(missing_name + '_' + str(ii + 1))
            else:
                first = x[ii].split('-')[0]  # first element is key
                xf = re.search('[0-9]*[\.|\,]*[0-9]+', str(first))
                if not xf:
                    # first = x[ii].split('/')  # first element is key
                    # zkey = [item for sublist in first for item in sublist]
                    z = x[ii:]
                    # z.append(zkey)
                    key_found = True
                    # Check if item is related to climate, energy, access
                    if x[ii] in ['climate', 'access', 'energy']:
                        energy_found = True
                    elif x[ii] in ['dtt']:
                        elem_found = True
                        elem_cod = z[1]
                        elem_cod_list.append(elem_cod)
                    
            ii += 1
            
        # Remove empty strings
        z = list(filter(None, z))

        if site_found and key_found:
            # print(z)
            z[-2] = z[-2].removeprefix(site_cod)
        if elem_found and key_found:
            z[-2] = z[-2].removesuffix(elem_cod)
        elem_cod_list = list(set(elem_cod_list))
        # Process Energy related alarms
        if energy_found and key_found:
            z_params = z[1:-1] # [z[1], z[-1]]
            z_serv_ini = z[0] # 'energy'
        elif energy_found:
            z_params = z[0:-2] # [z[1], z[-1]]
            z_serv_ini = z[-1] # 'energy'
        else:
            if monitor_found | transmitter_found:
                # Extract params
                z_params = [z[1]]
                z_serv_ini = z[-1].removeprefix('service_') # z[-1].split('_')[-1]
            elif key_found:
                # Extract params
                z_params = z[3:-2]
                z_serv_ini = z[-1]

    elif pat_x == 1: # 'RF - CH - Service'
        z = x_dash
        z_serv_ini = x_dash[-1]
        z_params = x_dash[0:2]
        
    z_serv = 'unknown'
    # Check if it exists in serv_equiv_dictionary
    if z_serv_ini is not None:
        if len(z_serv_ini.split('-')) == 1:
            for key, value in serv_equiv_dict.items():
                if z_serv_ini == key:
                    z_serv = key
                elif z_serv_ini in value:
                    z_serv = key
        else:
            z_serv = ", ".join(z_serv_ini.split('-'))
        
        if z_params is not None:
            if len(z_params)==1:
                z_params = z_params[0]
            else:
                z_params = ", ".join(z_params)
            z_params = z_params.replace('_', ' ')
    return ", ".join(z).replace('_', ' '), z_params, z_serv
# --------------------------------------------------------------------------------------------------

def discretize_views_impct(x, sep_value):
    """ Discretize impact of X as no, medium high given given a SEP_VALUE """
    if x == 0:
        return 'no'
    elif (x > 0) and (x <= sep_value):
        return 'medium'
    else:
        return 'high'

# --------------------------------------------------------------------------------------------------


def discretize_service_impct(x, values_list):
    """ Discretize service impact (X) through binning expressed by VALUES_LIST list """

    if isinstance(x,str):
        return x
    else:
        values_list.sort()
        if x <= values_list[0]:
            # return 'impact_00'
            return 'impact00'
        if x > values_list[-1]:
            # return 'impact_' + str(len(values_list)).zfill(2)
            return 'impact' + str(len(values_list)).zfill(2)
        for ii, value in enumerate(values_list):
            if (x <= values_list[ii + 1]) and (x > values_list[ii]):
                # return 'impact_' + str(ii + 1).zfill(2)
                return 'impact' + str(ii + 1).zfill(2)
            else:
                continue

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------


def clean(s, mask_num_flag=0):
    """ Remove punctuation and clean lines as text sentences

    Args:
        s               (str) input text line
        mask_num_flag   (int) flag for removing (0), masking with * (1), keeping (2) numeric strings, except IP

    Returns:
        s (str) output, cleaned text line

    """

    s = re.sub(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', 'ip', s)  # replace IP numbers for ip word
    s = s.translate(str.maketrans(string.punctuation,
                                  ' ' * len(string.punctuation)))  # separate strings when punctuation is present
    s = s.lower()  # lower the characters
    s = re.sub('ñ', 'n', s)  # remove spanish ñ
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])  # split words
    s = " ".join(
        [" ".join(re.split(r'\s+|(\d+)', word)) for word in s.strip().split()])  # split subwords for isolating numbers
        
    if mask_num_flag==0: # remove numbers except IP

        s = " ".join([re.sub('^[(\d)*]((\d){0,6}$)|^[(\d)*]((\d){9,20}$)', '', word) for word in
                      s.strip().split()])  # mask all numbers except 8bits parts: IPs
    elif mask_num_flag==1: # mask numbers with generic * (except IP)
        s = " ".join([re.sub('^[(\d)*]((\d){0,6}$)|^[(\d)*]((\d){9,20}$)', '*', word) for word in
                      s.strip().split()])  # mask all numbers except 8bits parts: IPs
    else: # keep numbers
        pass # do nothing in other cases
    s = re.sub(' +', ' ', s)
    # note this last line may treat as "words" other potential 8digit strings.

    return s

# --------------------------------------------------------------------------------------------------


def clean_UC3M(s, mask_num_flag=0):
    """ Remove punctuation and clean lines as text sentences

    Args:
        s               (str) input text line
        mask_num_flag   (int) flag for removing (0), masking with * (1), keeping (2) numeric strings, except IP

    Returns:
        s (str) output, cleaned text line

    """

    s = re.sub(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', 'ip', s)  # replace IP numbers for ip word
    s = s.translate(str.maketrans(string.punctuation,
                                  ' ' * len(string.punctuation)))  # separate strings when punctuation is present
    s = s.lower()  # lower the characters
    s = re.sub('ñ', 'n', s)  # remove spanish ñ
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])  # split words

    if mask_num_flag==0: # remove numbers except IP
        s = " ".join(
            [" ".join(re.split(r'\s+|(\d+)', word)) for word in s.strip().split()])  # split subwords for isolating numbers
        s = " ".join([re.sub('^[(\d)*]((\d){0,6}$)|^[(\d)*]((\d){9,20}$)', '', word) for word in
                      s.strip().split()])  # mask all numbers except 8bits parts: IPs
    elif mask_num_flag==1: # mask numbers with generic * (except IP)
        s = " ".join(
            [" ".join(re.split(r'\s+|(\d+)', word)) for word in s.strip().split()])  # split subwords for isolating numbers
        s = " ".join([re.sub('^[(\d)*]((\d){0,6}$)|^[(\d)*]((\d){9,20}$)', '*', word) for word in
                      s.strip().split()])  # mask all numbers except 8bits parts: IPs
    else: # keep numbers
        pass # do nothing in other cases
    s = re.sub(' +', ' ', s)
    # note this last line may treat as "words" other potential 8digit strings.

    return s

# --------------------------------------------------------------------------------------------------