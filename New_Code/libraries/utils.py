# -*- coding: utf-8 -*-

# copyright GMV 2022
# created by ggnn

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import os
import yaml
from shutil import copyfile
import sys
import logging
import re
from datetime import datetime as dt

from libraries.configRead import get_config

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


# ----------------------------------------------------------------------------------------------------------------------
def prepare_sess_dvc(config_task_filename, task_folder, task_fun_filename,  env_dict, dvc_flag=True, sess_date=None):
    """Prepare embeddings projector session.

    Args:
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths
        dvc_flag                (bool)   flag for preparing dvc run writing to "swap" folder or locally save output

    Returns:
        this_run_folder:       (string) folder with main outcome of the task

    """

    logger.info('Session preparation')

    # Create directory to save intermediate models
    try:
        config_dict =  yaml.safe_load(open(os.path.join(env_dict["config_path"],config_task_filename)))
    except OSError:
        logger.error("   Cannot find configuration file: {}".format(config_task_filename))
        logger.error("Current Task has been ABORTED")
        sys.exit()

    # output parsing
    env_path_name_is_list = False
    out_path_is_list = False
    env_path_name = config_dict['outs']['env_path_name']
    if isinstance(env_path_name, list):
        env_path_name_is_list = True
        env_path = []
        for name in env_path_name:
            env_path.append(env_dict[name])
    else:
        env_path = env_dict[env_path_name]
    out_path = config_dict['outs']['output_folder'] # pay attention, in this case out_path is a list
    if isinstance(out_path, list):
        out_path_is_list = True
    if ((env_path_name_is_list and out_path_is_list) and (len(env_path_name)!=len(out_path)) or \
        (env_path_name_is_list and not out_path_is_list) or \
        (not env_path_name_is_list and out_path_is_list)):
        logger.error("env_path_name and output_folder must have same number of elements")
        return

    if dvc_flag: # RUN WITH DVC - WRITE THE DVC.JSON FILE WITH DEPENDENCIES
        if env_path_name_is_list:
            this_run_folder = []
            for ii, this_env_path in enumerate(env_path):
                this_run_folder.append(os.path.join('swap', env_path[ii], out_path[ii]))
                if not os.path.exists(this_run_folder[ii]):
                    logger.info("   New folder created: {}".format(this_run_folder[ii]))
                    os.makedirs(this_run_folder[ii])
            out_folder = this_run_folder[0]

        else:
            this_run_folder = os.path.join('swap', env_path, out_path)
            if not os.path.exists(this_run_folder):
                logger.info("   New folder created: {}".format(this_run_folder))
                os.makedirs(this_run_folder)
            out_folder = this_run_folder

        # it should use here:
        #       task folder and  task filename identified as dep
        #       config file
        #       those params inside "deps" subsection (files)

    else: # RUN WITHOUT DVC - SAVE LOCALLY A COPY OF MAIN DEPENDENCIES (config and code)
        if env_path_name_is_list:
            this_run_folder = []
            for ii, this_env_path in enumerate(env_path):
                if ii > 0: # for folders others than first (index-0), use name of first element of out_path list for final name
                    this_run_folder.append(os.path.join( env_path[ii], out_path[ii], out_path[0] + '_' + sess_date))
                else:
                    this_run_folder.append(os.path.join(env_path[ii], out_path[ii], out_path[ii] + '_' + sess_date))
                if not os.path.exists(this_run_folder[ii]):
                    logger.info("   New folder created: {}".format(this_run_folder[ii]))
                    os.makedirs(this_run_folder[ii])
            copyfile(os.path.join(env_dict["config_path"], config_task_filename),
                     os.path.join(this_run_folder[0], config_task_filename))
            copyfile(os.path.join(sys.path[0], task_folder, task_fun_filename),
                     os.path.join(this_run_folder[0], task_fun_filename))
            logger.info("   Task configuration file copied to " + os.path.join(this_run_folder[0], config_task_filename))
            logger.info("   Task function copied to " + os.path.join(this_run_folder[0], task_fun_filename))
            out_folder = this_run_folder[0]
        else:
            this_run_folder = os.path.join( env_path, out_path, out_path + '_' + sess_date)
            if not os.path.exists(this_run_folder):
                logger.info("   New folder created: {}".format(this_run_folder))
                os.makedirs(this_run_folder)
            copyfile(os.path.join(env_dict["config_path"], config_task_filename),
                     os.path.join(this_run_folder, config_task_filename))
            copyfile(os.path.join(sys.path[0], task_folder, task_fun_filename),
                     os.path.join(this_run_folder, task_fun_filename))
            logger.info("   Task configuration file copied to " + os.path.join(this_run_folder, config_task_filename))
            logger.info("   Task function copied to " + os.path.join(this_run_folder, task_fun_filename))
            out_folder = this_run_folder

    return out_folder


# ----------------------------------------------------------------------------------------------------------------------
def prepare_emb_projector(config_task_filename, task_folder, task_fun_filename, sess_date,  env_dict):
    """Prepare embeddings projector session.

    Args:
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths

    Returns:
        this_run_folder:        (string) tensorbooard log folder containing viewer data

    """

    logger.info('Session preparation')

    # Create directory to save intermediate models
    try:
        config_dict =  yaml.safe_load(open(os.path.join(env_dict["config_path"],config_task_filename)))
    except OSError:
        logger.error("   Cannot find configuration file: {}".format(config_task_filename))
        logger.error("Current Task has been ABORTED")
        sys.exit()

    tb_logs_folder = os.path.join(env_dict['output_path'], 'tb_embs_viewer')
    this_run_folder = os.path.join(tb_logs_folder, 'tb_embs_viewer_'+sess_date)
    if not os.path.exists(this_run_folder):
        logger.info("   New folder created: {}".format(this_run_folder))
        os.makedirs(this_run_folder)

    copyfile(os.path.join(env_dict["config_path"],config_task_filename),
             os.path.join(this_run_folder, config_task_filename))
    copyfile(os.path.join(sys.path[0],task_folder,task_fun_filename),
             os.path.join(this_run_folder, task_fun_filename))

    logger.info("   Task configuration file copied to " + os.path.join(this_run_folder, config_task_filename))
    logger.info("   Task function copied to " + os.path.join(this_run_folder, task_fun_filename))

    return this_run_folder


# ----------------------------------------------------------------------------------------------------------------------
def prepare_sess(config_task_filename, task_folder, task_fun_filename, sess_date,  env_dict, env_folder, outfolder_name):
    """Prepare embeddings projector session.

    Args:
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths
        env_folder:             (string) name of the environment folder for output (interim, model, output)
        outfolder_name:         (string) output folder name for the preprocess session (e.g. merged_data)

    Returns:
        tb_log_dir:      (string) tensorbooard log folder containing viewer data

    """

    logger.info('Session preparation')

    # Create directory to save intermediate models
    try:
        config_dict =  yaml.safe_load(open(os.path.join(env_dict["config_path"],config_task_filename)))
    except OSError:
        logger.error("   Cannot find configuration file: {}".format(config_task_filename))
        logger.error("Current Task has been ABORTED")
        sys.exit()

    task_output_folder = os.path.join(env_dict[env_folder], outfolder_name)
    this_run_folder = os.path.join(task_output_folder, outfolder_name+'_'+sess_date)
    if not os.path.exists(this_run_folder):
        logger.info("   New folder created: {}".format(this_run_folder))
        os.makedirs(this_run_folder)

    copyfile(os.path.join(env_dict["config_path"],config_task_filename),
             os.path.join(this_run_folder, config_task_filename))
    copyfile(os.path.join(sys.path[0],task_folder,task_fun_filename),
             os.path.join(this_run_folder, task_fun_filename))

    logger.info("   Task configuration file copied to " + os.path.join(this_run_folder, config_task_filename))
    logger.info("   Task function copied to " + os.path.join(this_run_folder, task_fun_filename))

    return this_run_folder


# ----------------------------------------------------------------------------------------------------------------------
def prepare_prepro_sess(config_task_filename, task_folder, task_fun_filename, sess_date,  env_dict, outfolder_name):
    """Prepare embeddings projector session.

    Args:
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths
        outfolder_name:         (string) output folder name for the preprocess session (e.g. merged_data)

    Returns:
        tb_log_dir:      (string) tensorbooard log folder containing viewer data

    """

    logger.info('Session preparation')

    # Create directory to save intermediate models
    try:
        config_dict =  yaml.safe_load(open(os.path.join(env_dict["config_path"],config_task_filename)))
    except OSError:
        logger.error("   Cannot find configuration file: {}".format(config_task_filename))
        logger.error("Current Task has been ABORTED")
        sys.exit()

    task_output_folder = os.path.join(env_dict['interim_path'], outfolder_name)
    this_run_folder = os.path.join(task_output_folder, outfolder_name+'_'+sess_date)
    if not os.path.exists(this_run_folder):
        logger.info("   New folder created: {}".format(this_run_folder))
        os.makedirs(this_run_folder)

    copyfile(os.path.join(env_dict["config_path"],config_task_filename),
             os.path.join(this_run_folder, config_task_filename))
    copyfile(os.path.join(sys.path[0],task_folder,task_fun_filename),
             os.path.join(this_run_folder, task_fun_filename))

    logger.info("   Task configuration file copied to " + os.path.join(this_run_folder, config_task_filename))
    logger.info("   Task function copied to " + os.path.join(this_run_folder, task_fun_filename))

    return this_run_folder


# ----------------------------------------------------------------------------------------------------------------------
def prepare_dict_sess(config_task_filename, task_folder, task_fun_filename, sess_date,  env_dict, outfolder_name):
    """Prepare embeddings projector session.

    Args:
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths
        outfolder_name:         (string) output folder name for the preprocess session (e.g. merged_data)

    Returns:
        tb_log_dir:      (string) tensorbooard log folder containing viewer data

    """

    logger.info('Session preparation')

    # Create directory to save intermediate models
    try:
        config_dict =  yaml.safe_load(open(os.path.join(env_dict["config_path"],config_task_filename)))
    except OSError:
        logger.error("   Cannot find configuration file: {}".format(config_task_filename))
        logger.error("Current Task has been ABORTED")
        sys.exit()

    task_output_folder = os.path.join(env_dict['dict_path'], outfolder_name)
    this_run_folder = os.path.join(task_output_folder, outfolder_name+'_'+sess_date)
    if not os.path.exists(this_run_folder):
        logger.info("   New folder created: {}".format(this_run_folder))
        os.makedirs(this_run_folder)

    copyfile(os.path.join(env_dict["config_path"],config_task_filename),
             os.path.join(this_run_folder, config_task_filename))
    copyfile(os.path.join(sys.path[0],task_folder,task_fun_filename),
             os.path.join(this_run_folder, task_fun_filename))

    logger.info("   Task configuration file copied to " + os.path.join(this_run_folder, config_task_filename))
    logger.info("   Task function copied to " + os.path.join(this_run_folder, task_fun_filename))

    return this_run_folder


# ----------------------------------------------------------------------------------------------------------------------
def prepare_scaled_corr_gen_session(uc_path, config_task_filename, task_folder, task_fun_filename, sess_date, env_dict):
    """Prepare scaled-correlation generation session.

    Args:
        uc_path:                (string) use case relative path name
        config_task_filename:   (string configuration file name (.ini)
        task_folder:            (string) task folder
        task_fun_filename:      (string) task function name (.py)
        sess_date:              (string) session date
        env_dict:               (dict)   dict containing environment paths

    Returns:
        embeddings folder:      (string) embeddings folder

    """
    # Create directory to save intermediate models
    logger.info("SCALED-CORRELATIONS GENERATION session preparation")
    config_dict = get_config(os.path.join(uc_path, env_dict['config_path'], config_task_filename))
    print(config_dict.keys())
    scaled_corr_folder = os.path.join(uc_path,
                                      env_dict["scorr_path"],
                                      'scorr_' + sess_date + '_'+
                                      '{}_feat_{}'.format(re.sub('raw_','',
                                                                 config_dict['raw_data_path'],
                                                                 flags=re.IGNORECASE),
                                                          config_dict['col_of_interest_file'].
                                                          rstrip('txt').rstrip('.').
                                                          split('_')[-1]))
    if not os.path.exists(scaled_corr_folder):
        os.makedirs(scaled_corr_folder)
        logger.info("   Directory " + scaled_corr_folder + " created")
    else:
        logger.info("   Directory " + scaled_corr_folder + " already exists")

    copyfile(os.path.join(uc_path,env_dict["config_path"],config_task_filename),
             os.path.join(scaled_corr_folder, config_task_filename))
    copyfile(os.path.join(sys.path[0],'esa_aimgnss',uc_path.lower(),task_folder,task_fun_filename),
             os.path.join(scaled_corr_folder, task_fun_filename))

    logger.info("   Task configuration file copied to " + os.path.join(scaled_corr_folder, config_task_filename))
    logger.info("   Task function copied to " + os.path.join(scaled_corr_folder, task_fun_filename))

    return scaled_corr_folder


# ----------------------------------------------------------------------------------------------------------------------
def close_session(log_filename, sess_folder):
    """Close session.
    Copy the log file to the session folder.

    Args:
        log_filename: log filename with extension (string)
        sess_folder:  session folder name(string)

    Returns:
        0: correctly closed session

    """

    logger.info('Session closure')
    if os.path.exists(log_filename):
        copyfile(log_filename, os.path.join(sess_folder, log_filename))
    return 0


# ----------------------------------------------------------------------------------------------------------------------
def mt_difference(myarray,myinterval=1):
    """
    Helper function computing a time difference for multivariate time series (mt) in form of array (myarray).
    The time difference is computed between last value and (myinterval) number of steps in the past.
    For the sake of generality if the multivariate array only contains scalars instants (cross-sectional) values,
    also this special case can be managed.

    :param myarray: (numpy array) array containing multivariate time series, variables distributed by rows, time indices distributed by column
    :param myinterval: (integer) number of steps for considering a delta in time
    :return: outarray: (numpy array) difference multivariate time series array
            maskarray: (numpy array) array mask containing a -1 dummy values in places where dummy values exist. Other elements are set to zero

    """
    if myarray.shape[1]-myinterval > 0:
        vect = np.arange(myarray.shape[1]-myinterval,step=1)
        outarray = np.zeros((myarray.shape[0],len(vect)))
        maskarray = np.ones((myarray.shape[0],len(vect)))
        for ii in vect:
            for jj in np.arange(myarray.shape[0]):
                if myarray[jj,ii+myinterval]==-1 or myarray[jj,ii]==-1:
                    outarray[jj,ii] = -1
                    maskarray[jj,ii]= -1
                else:
                    outarray[jj,ii]  = myarray[jj,ii+myinterval]-myarray[jj,ii]
    else:
        outarray = np.zeros((myarray.shape[0],1))
        maskarray = np.ones((myarray.shape[0],1))
        for jj in np.arange(myarray.shape[0]):
            if myarray[jj,0]==-1 or myarray[jj,0]==-1:
                outarray[jj,0] = -1
                maskarray[jj,0] = -1
            else:
                outarray[jj,0]  = myarray[jj,myinterval-1]-myarray[jj,0]
#         outarray[:,0] = myarray[:,myinterval-1]-myarray[:,0]
    return outarray, maskarray


# ----------------------------------------------------------------------------------------------------------------------
def subtract_station_pos_tf(batch,batch_shape,stat_json,stations_indices,remove_satpos=False):
    """
    This function subtracts the station position to the satellite position in order to debias the value"

    :param batch:         (TensorFlow batch) batch of models data samples
    :param batch_shape:   (integer tuple) shape of the batch, as (n_batch,n_var,n_times)
    :param stat_json:     (json)    json structure containing stations information
    :param stations_indices: (list) list of integers indices for station columns identifiers
    :param remove_satpos: (boolean) Boolean for removing absolute satellite position
    :return: new_batch: (TensorFlow batch) batch of modified data samples

    """
    rho = np.zeros((len(stations_indices),batch_shape[2]))
    new_batch = np.zeros((batch_shape[0],batch_shape[1]+len(stations_indices),batch_shape[2]))
    for ss in np.arange(batch_shape[0]): # for every sample in the batch
        ii = 0
        for st_id in stations_indices:
            for tt in np.arange(batch_shape[2]): # for every time step
                if (batch[ss,st_id,tt] > 0):
                    rho[ii,tt] = np.linalg.norm(batch[ss,[1,2,3],tt]-
                                                [stat_json[batch[ss,st_id,tt].astype(int)-1]['POS_X[m]'],
                                                 stat_json[batch[ss,st_id,tt].astype(int)-1]['POS_Y[m]'],
                                                 stat_json[batch[ss,st_id,tt].astype(int)-1]['POS_Z[m]']]
                                                 ,axis=0)
                else:
                    rho[ii,tt] = -1
            ii = ii + 1
        new_batch[ss,:,:] = np.insert(batch[ss,:,:],stations_indices+1,rho,axis=0)
    if remove_satpos:
        new_batch = np.delete(new_batch,[1,2,3],axis=1)
    return new_batch

# ----------------------------------------------------------------------------------------------------------------------


def downsample(mode,decimation,N_original):
    """
    This function, given an integer number of samples (N_original),
    a decimation rescaling factor (decimation),
    and mode (Mode: linear, logarithm, chebyshev nodes distribution),
    returns a vector of downsampled index following (approximately) the desired
    downsampling distribution.

    :param mode:        (string) decimation distribution/mode
    :param decimation:  (integer) decimation factor (1 means no decimation)
    :param N_original:  (integer) number of original nodes
    :return:            (list of integers) vector of decimated indices

    """
    if decimation!=1:
        N = int(N_original/decimation)
        if mode == 'lin':
            idx=np.arange(0,N_original,decimation,dtype=int)
        elif mode == 'log':
            idx=np.sort(N_original-np.unique(np.logspace( np.log10(1), np.log10(N_original),base=10,num=N).astype(int)))
        elif mode == 'cheb':
            n = np.arange(0,N)
            idx = (0+N_original)/2 + (N_original-0)/2*np.cos(((2.*(n)-1)/(2*N))*np.pi)
            idx  = np.unique(idx.astype(int))
        else:
            idx=np.arange(0,N_original,decimation,dtype=int)
    else:
        idx = np.arange(0,N_original,dtype=int)
    return idx


# ----------------------------------------------------------------------------------------------------------------------
def get_quantile_based_boundaries(feature_values, num_buckets):
  """Get quanatiles of features distributions based on number of buckets.

  Args:
      feature_values:  dict of features
      num_buckets:     number of buckets

  Returns:
        quantiles (list) quantiles for each features key
  """
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]


# ----------------------------------------------------------------------------------------------------------------------
def get_num_samples(n_files, n_lines, block_length, window_length, win_shift):
    """Estimate number of samples when a sliding windows approach is applied to a dataset.

    Args:
        n_files:        (int) number of files
        n_lines:        (int) number of lines in a file
        block_length:   (int) number of lines read in a block from a file
        window_length:  (int) size of a window
        win_shift:      (int) shift of the sliding window

    Returns:
        n_samples (int) estimated number of samples in the dataset (approx)

    """
    # nun_samples = n_file * n_blocks in a file * n windows in a block
    # return n_files*int(np.floor((n_lines/block_length)))*(int(np.floor((block_length-window_length)/win_shift))+1)
    return n_files*int(np.floor((n_lines/block_length)))*(int(np.floor((block_length-window_length+win_shift)/win_shift)))


# ----------------------------------------------------------------------------------------------------------------------
def get_closest_file_by_date(list_of_files, ref_date):
    """Return the filename with the closest date to the reference date.

    Given a list of file names [LIST_OF_FILES] and a reference data as string [REF_DATE] in format YYYYMMDD
    this function return the closest file by date. It is assumed the file files are named with same convention.

    Args:
        list_of_files: (list of strings) list of filenames including date YYYYMMDD, it can include the full path
        ref_date:      (string) reference date YYYYMMDD

    Returns:
        closest_file_by_date (string) closest file by date, it can include full path according to models
    """

    # for the sake of being sure, sort the list by name.
    # Naming convention is assumed following YYYYDDMM
    list_of_files.sort()
    closest_file_by_date = list_of_files[0]
    ref_dt = dt.strptime(ref_date, '%Y%m%d')
    min_diff = np.inf
    for filename in list_of_files:
        file_date = re.match('.+(\d\d\d\d\d\d\d\d)', filename).group(1)
        file_dt = dt.strptime(file_date, '%Y%m%d')
        this_diff = np.abs((ref_dt-file_dt).days)
        if this_diff <= min_diff:
            min_diff = this_diff
            closest_file_by_date = filename

    return closest_file_by_date


# ----------------------------------------------------------------------------------------------------------------------
def get_closest_file_by_date_v2(list_of_files, ref_date):
    """Return the filename with the closest date to the reference date.

    Given a list of file names [LIST_OF_FILES] and a reference data as string [REF_DATE] in format YYYYMMDD
    this function return the closest file by date. It is assumed the file files are named with same convention.

    Args:
        list_of_files: (list of strings) list of filenames including date YYYYMMDD, it can include the full path
        ref_date:      (string) reference date YYYYMMDD

    Returns:
        closest_file_by_date (string) closest file by date, it can include full path according to models
    """

    # for the sake of being sure, sort the list by name.
    # Naming convention is assumed following YYYYDDMM
    list_of_files.sort()
    closest_file_by_date = list_of_files[0]
    ref_dt = dt.strptime(ref_date, '%Y%m%d')
    min_diff = np.inf
    for filename in list_of_files:
        file_date =  re.match('.+(\d\d\d\d)_(\d\d)_(\d\d)', filename).group(1)+ \
                     re.match('.+(\d\d\d\d)_(\d\d)_(\d\d)', filename).group(2)+ \
                     re.match('.+(\d\d\d\d)_(\d\d)_(\d\d)', filename).group(3)
        file_dt = dt.strptime(file_date, '%Y%m%d')
        this_diff = np.abs((ref_dt-file_dt).days)
        if this_diff <= min_diff:
            min_diff = this_diff
            closest_file_by_date = filename

    return closest_file_by_date


# ----------------------------------------------------------------------------------------------------------------------
def generate_file_list_part(sat_list):
    """Generate part of string name, using the satellite lists to be considered.
    all -->  all satellite considered
    01 --> only one satellite (01) is considered
    subset --> if more than two, but not all satellites are considered

    Args:
        sat_list: (list) list of satellites IDs

    Returns:
        sat_part: (str) string for defining dataset subset name

    """
    if len(sat_list)==0:
        sat_part = '_all_'
    elif len(sat_list)==1:
        sat_part = '_E'+str(sat_list[0]).zfill(2)+'_'
    else:
        sat_part = '_subset_'
    return sat_part


# ----------------------------------------------------------------------------------------------------------------------
def generate_folder_name(uc, data_folders, sat_list):
    """Generate a new folder name, given the list of satellites ID (SAT_LIST) to consider, the use case (UC), and the
    list of folders containing the dataset.

    Args:
        uc:             (int) use case
        data_folders:   (list of str) list of data folders
        sat_list:       (list) list of satelites IDs

    Returns:
        folder_name     (str) generated folder name

    """
    if len(sat_list)==0:
        sat_part = '_all_'
    elif len(sat_list)==1:
        sat_part = '_E'+str(sat_list[0]).zfill(2)+'_'
    else:
        sat_part = '_subset_'

    min_datetime = dt.strptime('22000101', '%Y%m%d')
    max_datetime = dt.strptime('19000101', '%Y%m%d')
    for folder in data_folders:
        dates = re.findall('(\d\d\d\d\d\d\d\d)',folder)
        for this_date in dates:
            this_datetime = dt.strptime(this_date, '%Y%m%d')
            if this_datetime <= min_datetime:
                min_datetime = this_datetime
            if this_datetime >= max_datetime:
                max_datetime = this_datetime
    if min_datetime == max_datetime:
        folder_name = str(uc) + sat_part + min_datetime.strftime('%Y%m%d')
    else:
        folder_name = str(uc) + sat_part + min_datetime.strftime('%Y%m%d') + '_' + max_datetime.strftime('%Y%m%d')

    return folder_name


# ----------------------------------------------------------------------------------------------------------------------
def invTransform(scaler, data, colName, colNames):
    """Given a scaler (SCALER) object from scikit-learn transformations, this function perform the inverse transformation
    on a specific column name of the full numerical features matrix.

    Args:
        scaler:     (scikit-object) scaler object from scikit learn
        data:       (np.array)  scaled models features to be inversely-scaled
        colName:    (str) name of the feature of interest
        colNames:   (list) list of numerical columns in data, must have len = data.shape[1]

    Returns:
                    (np.array) the inversly-scaled feature as numpy array of only one dimension.

    """
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName]=data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


# ----------------------------------------------------------------------------------------------------------------------
def convert_int(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

