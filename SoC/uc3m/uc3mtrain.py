#!/usr/bin/esnv python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:12 2023

@author: SOCTRA Team
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries 
import pandas as pd
import numpy as np
import logging
import pickle

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from uc3m.mlpbayesian import MLPBayes, MLPBayesBin, MLPBayesBinW, MLPAutoencoder, MLPBayesBinMT, MLPOrdinal
from uc3m.labelswitching import LSEnsemble

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_predict_nfold(coded_data, classifiers, parameters, 
                       num_learners=11, use_NLP=True,
                       hypertuning=True, scoring_metric='balanced_accuracy',
                       labels = ['False', 'True'],out_name=None):
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::                     
    #--------------------------------------------------------------------------
    # Organizing coded data
    #--------------------------------------------------------------------------        
    x_train_list, x_test_list, y_train_list, y_test_list, idx_test_list, coding_labels, nlp_feats, df_train_list, df_test_list, binary_type, normalization, text_enc= coded_data
   
    if isinstance(x_train_list, list):
        num_folds = len(x_train_list)
    else:   # num_folds = 1
        num_folds = 1
        x_train_list = [x_train_list]
        x_test_list = [x_test_list]
        y_train_list = [y_train_list]
        y_test_list = [y_test_list]
        df_train_list = [df_train_list]
        df_test_list = [df_test_list]
    
    df_alarms = pd.concat([df_train_list[0], df_test_list[0]]).sort_index()
                
    cw_train = df_train_list[0]['weight'].to_numpy()        
    #--------------------------------------------------------------------------
    # Model training and evaluation
    #--------------------------------------------------------------------------
    classifiers =[eval(x) for x in classifiers]
    model_names = []
    res_model = []
    #Iterate over classifiers
    for model, parameters_model in zip(classifiers, parameters):
        model_name = model.__class__.__name__
        logger.info("Calculating results for {}".format(model_name))
        model_names.append(model_name)
        
        y_test_tot = np.zeros((df_alarms.shape[0],))
        ye_test_tot = np.zeros((df_alarms.shape[0],))
        
        
        for kFold in range(num_folds):        
            
            logger.info("  * Processing Fold {}".format(kFold+1))
                                                
            x_train = x_train_list[kFold]
            x_test = x_test_list[kFold] 
            y_train = y_train_list[kFold]
            y_test = y_test_list[kFold]
                        
            if use_NLP == False:
                x_train = x_train[:,0:len(coding_labels[kFold])]
                x_test = x_test[:,0:len(coding_labels[kFold])]
            
            logger.info("    + Size of training data : {}".format(x_train.shape))
            logger.info("    + Size of test data     : {}".format(x_test.shape))
            
            # y_train = 2*y_train-1
            # y_test = 2*y_test-1
            
            ye_test_fold = np.zeros((y_test.shape[0],num_learners))
            
            for kRep in range(num_learners):
                logger.info('        * Training classifier - Fold {}/{} : Learner {} of {}'.format(kFold+1, num_folds, kRep+1, num_learners))
                
                if hypertuning:
                  model_CV = GridSearchCV(model, parameters_model, scoring=scoring_metric) 
                  model_CV.fit(x_train, y_train)
                  ye_test_fold[:,kRep] = model_CV.best_estimator_.predict(x_test)
                  
                else:
                  model.fit(x_train, y_train)
                  ye_test_fold[:,kRep] = model.predict(x_test)
            
            ye_test = np.median(ye_test_fold, axis=1).astype(int)
            
            TN, FP, FN, TP = confusion_matrix(y_test, ye_test).ravel()    
            logger.info(' ')
            logger.info(' Fold {}/{} : Acc = {}, Bal. Acc = {}'.format(kFold+1, num_folds, accuracy_score(y_test, ye_test), balanced_accuracy_score(y_test, ye_test)))
            logger.info(' Fold {}/{} : TN = {}, FP = {}, FN = {}, TP = {} '.format(kFold+1, num_folds, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
        
            
            y_test_tot[idx_test_list[kFold]] = y_test
            ye_test_tot[idx_test_list[kFold]] = ye_test
            
        if num_folds > 1:
            TN, FP, FN, TP = confusion_matrix(y_test_tot, ye_test_tot).ravel()
            logger.info(' ')
            logger.info(' {} : Acc = {}, Bal. Acc = {}'.format(model_name, accuracy_score(y_test_tot, ye_test_tot), balanced_accuracy_score(y_test_tot, ye_test_tot)))
            logger.info(' {} : TN = {}, FP = {}, FN = {}, TP = {} '.format(model_name, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
            
            print(' ')
            print('%20s & %1.5f & %1.5f & %1.5f & %1.5f & %1.5f \\\\'%(model_name, accuracy_score(y_test_tot, ye_test_tot),balanced_accuracy_score(y_test_tot, ye_test_tot),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)) )
            print(' ')
            
            cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test_tot, ye_test_tot), display_labels=labels)
            cm_display.plot()
            cm_display.ax_.set(xlabel='Predicted', ylabel='True', title='{} ({})'.format(model_name, scoring_metric))
            
        else:
            TN, FP, FN, TP = confusion_matrix(y_test, ye_test).ravel()
            logger.info(' ')
            logger.info(' {} : Acc = {}, Bal. Acc = {}'.format(model_name, accuracy_score(y_test, ye_test), balanced_accuracy_score(y_test, ye_test)))
            logger.info(' {} : TN = {}, FP = {}, FN = {}, TP = {} '.format(model_name, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
            
            print(' ')
            print('%20s & %1.5f & %1.5f & %1.5f & %1.5f & %1.5f \\\\'%(model_name, accuracy_score(y_test, ye_test),balanced_accuracy_score(y_test, ye_test),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)) )
            print(' ')
            
            cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, ye_test), display_labels=labels)
            cm_display.plot()
            cm_display.ax_.set(xlabel='Predicted', ylabel='True', title='{} ({})'.format(model_name, scoring_metric))
        
        
        
        val_pred = ye_test_tot == 1
        df_alarms.insert(loc=df_alarms.shape[1], column='Predict', value = val_pred)
        
        value_res =  df_alarms['Label'].astype(int).astype(str)+'-'+df_alarms['Predict'].astype(int).astype(str)    
        df_alarms.insert(loc=df_alarms.shape[1], column='Result', value=value_res)
        df_alarms.replace({'Result': {'0-1': 'False Alarm', '1-0':'No Detection', '0-0': 'Accurate', '1-1': 'Accurate'}},inplace=True)
                
        
        if num_folds > 1:
            res_model.append([model_name, accuracy_score(y_test_tot, ye_test_tot),balanced_accuracy_score(y_test_tot, ye_test_tot),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)])
        else:
            res_model.append([model_name, accuracy_score(y_test, ye_test),balanced_accuracy_score(y_test, ye_test),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)])
            df_alarms = df_alarms.loc[idx_test_list[kFold]]
                                                   
        
        if out_name != None:
            df_alarms.to_csv('{}_{}.csv'.format(out_name, model_name), index=False)
            logger.info('Predictions saved to {}'.format('{}_{}.csv'.format(out_name, model_name)))
    
                   
    return res_model, df_alarms


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_predict_nfold_W(coded_data, classifiers, parameters, 
                       num_learners=11, use_NLP=True,
                       hypertuning=True, scoring_metric='balanced_accuracy',
                       labels = ['False', 'True'],out_name=None):
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::                     
    #--------------------------------------------------------------------------
    # Organizing coded data
    #--------------------------------------------------------------------------        
    x_train_list, x_test_list, y_train_list, y_test_list, idx_test_list, coding_labels, nlp_feats, df_train_list, df_test_list, binary_type, normalization, text_enc= coded_data
   
    if isinstance(x_train_list, list):
        num_folds = len(x_train_list)
    else:   # num_folds = 1
        num_folds = 1
        x_train_list = [x_train_list]
        x_test_list = [x_test_list]
        y_train_list = [y_train_list]
        y_test_list = [y_test_list]
        df_train_list = [df_train_list]
        df_test_list = [df_test_list]
    
    df_alarms = pd.concat([df_train_list[0], df_test_list[0]]).sort_index()
                
    cw_train = df_train_list[0]['weight'].to_numpy()        
    #--------------------------------------------------------------------------
    # Model training and evaluation
    #--------------------------------------------------------------------------
    classifiers =[eval(x) for x in classifiers]
    model_names = []
    res_model = []
            
    input_size = x_train_list[0].shape[1]  # Number of features
                
    #Iterate over classifiers
    for model, parameters_model in zip(classifiers, parameters):
        
        model_name = model.__class__.__name__
        if model_name == 'LS_Ensemble':
            model.initialize_experts(input_size)  # Initialize the experts with input size
            
        logger.info("Calculating results for {}".format(model_name))
        model_names.append(model_name)
        
        y_test_tot = np.zeros((df_alarms.shape[0],))
        ye_test_tot = np.zeros((df_alarms.shape[0],))
        
        
        for kFold in range(num_folds):        
            
            logger.info("  * Processing Fold {}".format(kFold+1))
                                                
            x_train = x_train_list[kFold]
            x_test = x_test_list[kFold] 
            y_train = y_train_list[kFold].astype(int)
            y_test = y_test_list[kFold]
            # Convert y_test to integer
            y_test = y_test.astype(int)
            cw_train = df_train_list[kFold]['weight'].to_numpy()
            #cw_test = df_test_list[kFold]['weight'].to_numpy()            
            
                
            if use_NLP == False:
                x_train = x_train[:,0:len(coding_labels[kFold])]
                x_test = x_test[:,0:len(coding_labels[kFold])]
            
            logger.info("    + Size of training data : {}".format(x_train.shape))
            logger.info("    + Size of test data     : {}".format(x_test.shape))
                        
            ye_test_fold = np.zeros((y_test.shape[0],num_learners))
            
            for kRep in range(num_learners):
                logger.info('        * Training classifier - Fold {}/{} : Learner {} of {}'.format(kFold+1, num_folds, kRep+1, num_learners))
                
                if hypertuning:
                  model_CV = GridSearchCV(model, parameters_model, scoring=scoring_metric, n_jobs=1) 
                  model_CV.fit(x_train, y_train)
                  ye_test_fold[:,kRep] = model_CV.best_estimator_.predict(x_test)
                  
                  # Identify the best parameters found by GridSearchCV
                  best_params = model_CV.best_params_
                    
                  # Print or log the best parameters
                  logger.info(f"Best Parameters: {best_params}")
                  
                else:
                  model.fit(x_train, y_train, sample_weight=cw_train)
                  ye_test_fold[:,kRep] = model.predict(x_test)
            
            ye_test = np.median(ye_test_fold, axis=1).astype(int)
            
            TN, FP, FN, TP = confusion_matrix(y_test, ye_test).ravel()    
            logger.info(' ')
            logger.info(' Fold {}/{} : Acc = {}, Bal. Acc = {}'.format(kFold+1, num_folds, accuracy_score(y_test, ye_test), balanced_accuracy_score(y_test, ye_test)))
            logger.info(' Fold {}/{} : TN = {}, FP = {}, FN = {}, TP = {} '.format(kFold+1, num_folds, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
        
            
            y_test_tot[idx_test_list[kFold]] = y_test
            ye_test_tot[idx_test_list[kFold]] = ye_test
            
        if num_folds > 1:
            TN, FP, FN, TP = confusion_matrix(y_test_tot, ye_test_tot).ravel()
            logger.info(' ')
            logger.info(' {} : Acc = {}, Bal. Acc = {}'.format(model_name, accuracy_score(y_test_tot, ye_test_tot), balanced_accuracy_score(y_test_tot, ye_test_tot)))
            logger.info(' {} : TN = {}, FP = {}, FN = {}, TP = {} '.format(model_name, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
            
            print(' ')
            print('%20s & %1.5f & %1.5f & %1.5f & %1.5f & %1.5f \\\\'%(model_name, accuracy_score(y_test_tot, ye_test_tot),balanced_accuracy_score(y_test_tot, ye_test_tot),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)) )
            print(' ')
            
            cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test_tot, ye_test_tot), display_labels=labels)
            cm_display.plot()
            cm_display.ax_.set(xlabel='Predicted', ylabel='True', title='{} ({})'.format(model_name, scoring_metric))
            
        else:
            TN, FP, FN, TP = confusion_matrix(y_test, ye_test).ravel()
            logger.info(' ')
            logger.info(' {} : Acc = {}, Bal. Acc = {}'.format(model_name, accuracy_score(y_test, ye_test), balanced_accuracy_score(y_test, ye_test)))
            logger.info(' {} : TN = {}, FP = {}, FN = {}, TP = {} '.format(model_name, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
            
            print(' ')
            print('%20s & %1.5f & %1.5f & %1.5f & %1.5f & %1.5f \\\\'%(model_name, accuracy_score(y_test, ye_test),balanced_accuracy_score(y_test, ye_test),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)) )
            print(' ')
            
            cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, ye_test), display_labels=labels)
            cm_display.plot()
            cm_display.ax_.set(xlabel='Predicted', ylabel='True', title='{} ({})'.format(model_name, scoring_metric))
        
        
        
        val_pred = ye_test_tot == 1
        df_alarms.insert(loc=df_alarms.shape[1], column='Predict', value = val_pred)
        
        value_res =  df_alarms['Label'].astype(int).astype(str)+'-'+df_alarms['Predict'].astype(int).astype(str)    
        df_alarms.insert(loc=df_alarms.shape[1], column='Result', value=value_res)
        df_alarms.replace({'Result': {'0-1': 'False Alarm', '1-0':'No Detection', '0-0': 'Accurate', '1-1': 'Accurate'}},inplace=True)
                
        
        if num_folds > 1:
            res_model.append([model_name, accuracy_score(y_test_tot, ye_test_tot),balanced_accuracy_score(y_test_tot, ye_test_tot),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)])
        else:
            res_model.append([model_name, accuracy_score(y_test, ye_test),balanced_accuracy_score(y_test, ye_test),FP/(TN+FP), FN/(TP+FN), TP/(TP+FN)])
            df_alarms = df_alarms.loc[idx_test_list[kFold]]
                                                   
        
        if out_name != None:
            df_alarms.to_csv('{}_{}.csv'.format(out_name, model_name), index=False)
            logger.info('Predictions saved to {}'.format('{}_{}.csv'.format(out_name, model_name)))
    
                   
    return res_model, df_alarms


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_predict_nfold_AE(coded_data, n_neu = (250,), 
                         n_epoch=100, n_batch=150, 
                         multitask_labels=None,
                         num_learners=1, 
                         use_NLP=True, 
                         activations=['tanh','identity'],
                         binary_type = 'PlusMinusOne',
                         noise_type = 'gaussian',
                         noise_param = 0.1,
                         noise_gen = 'fixed',
                         dict_weight={0:1},
                         opt_weight = None,
                         opt_weight_AE='balanced',
                         drop_out = [], 
                         train_option='global',
                         out_model_path = None,
                         labels = ['False', 'True'],
                         out_name=None, 
                         flag_disp=False):
    """
    Perform n-fold prediction using an autoencoder-based model.

    This function uses an autoencoder-based model to perform n-fold cross-validation 
    prediction on the provided coded data.

    Parameters
    ----------
    coded_data : list
        The input data to be used for prediction.
        
    n_neu : tuple of int, optional
        Number of neurons in each hidden layer of the autoencoder. Default is (250,).
        
    n_epoch : int, optional
        Number of epochs for training. Default is 100.
        
    n_batch : int, optional
        Batch size for training. Default is 150.
        
    multitask_labels : list, optional
        Labels for multitask learning, if any. Default is None.
        
    num_learners : int, optional
        Number of learners to use. Default is 1.
        
    use_NLP : bool, optional
        Flag to indicate whether to use or not the NLP features. Default is True.
        
    activations : list of str, optional
        Activation functions to use in the layers. Default is ['tanh', 'identity'].
        
    binary_type : str, optional
        The type of binary encoding: 'PlusMinusOne' or 'ZeroOne'. Default is 'PlusMinusOne'.
        
    noise_type : str, optional
        Type of noise to add. Options are 'gaussian', 'salt_and_pepper', etc. Default is 'gaussian'.
        
    noise_param : float, optional
        Parameter controlling the amount of noise. Default is 0.1.
        
    noise_gen : str, optional
        Method of noise generation. Options are 'fixed', 'epoch'. Default is 'fixed'.
        
    dict_weight : dict, optional
        Dictionary specifying weights for different outputs. Default is {0: 1}.
        
    opt_weight : str, optional
        Option for weighting patterns. Options are 'operator' or None. Default is None.
        
    opt_weight_AE : str, optional
        Option for autoencoder weighting scheme for the reconstruction of the input.
        Options are: 
            + 'balanced': taking into account prior probabilities of main task
            + 'balanced_multi': considers all tasks (taking max value per pattern)
            + 'mixed': product of balance (main task) and operator weight
            + 'mixed_multi':  product of balance (all tasks) and operator weight        
        Default is 'balanced'.
        
    drop_out : list of float, optional
        Dropout rates for each layer. Default is an empty list.
        
    train_option : str, optional
        Training option to use. Options are 'global' or 'sequential'. Default is 'global'.
        
    out_model_path : str, optional
        Path to save the model. Default is None.
        
    labels : list of str, optional
        Labels for the output. Default is ['False', 'True'].
        
    out_name : str, optional
        Base of name of the output CSV file with results. Default is None.
        
    flag_disp : bool, optional
        Flag to indicate whether to display training progress. Default is False.

    Returns
    -------
    res_model: list
       Contains the weights of the different layers of the autoencoded
    df_alarms: dataframe
        Dataframe with the alarms that includes de decisions of the autoencoder
    
    """
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::                     
    #--------------------------------------------------------------------------
    # Organizing coded data
    #--------------------------------------------------------------------------        
    x_train_list, x_test_list, y_train_list, y_test_list, idx_test_list, coding_labels, nlp_feats, df_train_list, df_test_list, binary_type, normalization, text_enc= coded_data
   
    if isinstance(x_train_list, list):
        num_folds = len(x_train_list)
    else:   
        # For compatibility, the train and test data are converted in lists
        num_folds = 1
        x_train_list = [x_train_list]
        x_test_list = [x_test_list]
        y_train_list = [y_train_list]
        y_test_list = [y_test_list]
        df_train_list = [df_train_list]
        df_test_list = [df_test_list]
    
    df_alarms = pd.concat([df_train_list[0], df_test_list[0]]).sort_index()                    
    #--------------------------------------------------------------------------
    # Model training and evaluation
    #--------------------------------------------------------------------------
    model_name='AutoEncoder'
    res_model = []
    logger.info("Calculating results for AutoEncoder : {} neurons".format(n_neu))
    
    # Initialization of the total test labels and decision
    if not multitask_labels:
        multitask_labels = []
    
    y_test_tot = np.zeros((df_alarms.shape[0],len(multitask_labels)+1))
    ye_test_tot = np.zeros((df_alarms.shape[0],len(multitask_labels)+1))
    
    # Initiating the process per fold        
    for kFold in range(num_folds):        
        
        logger.info("  * Processing Fold {}".format(kFold+1))
        # Loading train and test data for the fold
        x_train = x_train_list[kFold]
        x_test = x_test_list[kFold] 
        y_train = y_train_list[kFold]
        y_test = y_test_list[kFold]
        # If NLP is False the text embeddings are removed from training data                                                                
        if use_NLP == False:
            x_train = x_train[:,0:len(coding_labels[kFold])]
            x_test = x_test[:,0:len(coding_labels[kFold])]            
        
        # Samples weights for a balanced approach (taking into account the label)
        class_counts = np.unique(y_train, return_counts=True)[1]
        ratio = class_counts[0] / class_counts[1]
        
        weights_balanced = np.ones(y_train.shape)
        weights_balanced[y_train == 1] = ratio
                        
        # Generation of the sample weights and output data to be reconstructed by the AE                
        idx_labels = [x for x in range(len(multitask_labels)+1)]
        idx_train = list(set(range(df_alarms.shape[0])) - set(idx_test_list[kFold]))
        
        # Generation of multask train and test labels
        y_train_mt = np.zeros((y_train.shape[0],len(multitask_labels)+1))        
        y_test_mt = np.zeros((y_test.shape[0],len(multitask_labels)+1))
        sample_weight = np.ones((x_train.shape[0],x_train.shape[1]+len(multitask_labels)+1))
        # The main label (train and test) is set in the first column
        y_train_mt[:,0] = y_train
        y_test_mt[:,0] = y_test        
        # Sample weight for the main label is set in the first column
        sample_weight[:,0] = weights_balanced 
        
        # Multitask labels and sample weights are included
        for count, label in enumerate(multitask_labels):
            if label == 'escalada':
                df_alarms['Label%d'%(count+1)] = (df_alarms['escalada_L2'] + df_alarms['escalada_procedimiento'])
            elif label in ['escalada_L2', 'escalada_procedimiento', 'reportado']:
                df_alarms['Label%d'%(count+1)] = df_alarms[label]
            
            y_train_mt[:,count+1] = df_alarms['Label%d'%(count+1)].loc[idx_train].astype(int).to_numpy()
            y_test_mt[:,count+1] = df_alarms['Label%d'%(count+1)].loc[idx_test_list[kFold]].astype(int).to_numpy()
            
            class_counts = np.unique(y_train_mt[:,count+1], return_counts=True)[1]
            ratio = class_counts[0] / class_counts[1]
            
            sample_weight[y_train_mt[:,count+1]==1,count+1] = ratio
        
        # Processing the option of weighting operators for multitask outputs
        weight_operator = df_alarms['weight'].loc[idx_train].to_numpy()
        if opt_weight == 'operator':                  
            for klabel in range(len(multitask_labels)+1):
                sample_weight[:,klabel] = sample_weight[:,klabel] * weight_operator

        # Processing the option of weighting for the reconstructed outputs
        if opt_weight_AE == 'balanced':
            sample_weight_seq = (dict_weight, weights_balanced)
            for kpat in range(x_train.shape[1]):
                sample_weight[:,kpat+len(multitask_labels)+1] = weights_balanced
        if opt_weight_AE == 'balanced_multi':
            weights_multi = np.max(sample_weight[:,0:len(multitask_labels)+1],axis=1)
            sample_weight_seq = (dict_weight, weights_multi)
            for kpat in range(x_train.shape[1]):
                sample_weight[:,kpat+len(multitask_labels)+1] = weights_multi
        elif opt_weight_AE == 'operator':
            sample_weight_seq = (dict_weight, weight_operator)
            for kpat in range(x_train.shape[1]):
                sample_weight[:,kpat+len(multitask_labels)+1] = weight_operator
        elif opt_weight_AE == 'mixed':
            sample_weight_seq = (dict_weight, weight_operator*weights_balanced)
            for kpat in range(x_train.shape[1]):
                sample_weight[:,kpat+len(multitask_labels)+1] = weight_operator * weights_balanced
        elif opt_weight_AE == 'mixed_multi':
            weights_multi = np.max(sample_weight[:,0:len(multitask_labels)+1],axis=1)
            sample_weight_seq = (dict_weight, weight_operator*weights_multi)
            for kpat in range(x_train.shape[1]):
                sample_weight[:,kpat+len(multitask_labels)+1] = weight_operator * weights_multi
        else:
            sample_weight_seq = (dict_weight,np.ones(x_train.shape[0]))
        
        for key in dict_weight.keys():
            sample_weight[:,key] = dict_weight[key] * sample_weight[:,key]
                
        x_out = np.append(y_train_mt,x_train,axis=1)
        ye_test_fold = np.zeros((num_learners,y_test.shape[0],len(multitask_labels)+1))            
                                                    
        # Labels and decision threshold given the specified binary format for labels
        if binary_type == 'PlusMinusOne':
            threshold_dec = 0
            x_out[:,idx_labels] = 2*x_out[:,idx_labels] - 1
        else:
            threshold_dec = 0.5
                
        logger.info("    + Size of training data : {}".format(x_train.shape))
        logger.info("    + Size of test data     : {}".format(x_test.shape))                        
        # Training of the AE learners
        for kLearn in range(num_learners):            
            logger.info('        * Training classifier - Fold {}/{} : Learner {} of {}'.format(kFold+1, num_folds, kLearn+1, num_learners))                
            if (len(n_neu)>1) & (train_option=='sequential'):
                # Sequential learning: Stacked De-noising AutoEncoders (SDAE)
                coefs_AE = []
                x_in_layer = x_train.copy()
                # Training of the initial AE layers
                for k in range(len(n_neu)-1):
                    if flag_disp:
                        logger.info('          x Training AE : Layer %d (%d neurons)'%(k+1,n_neu[k]))
                        
                    n_neu_layer=(n_neu[k],)
                    activations_layer=[activations[k],activations[k+1]]
                    drop_out_layer = [drop_out[k],drop_out[k+1]]
                    
                    model = MLPAutoencoder(layers_size=n_neu_layer, n_epoch=n_epoch[0], n_batch=n_batch,
                                           noise_type=noise_type, noise_param=noise_param, noise_gen=noise_gen,
                                           activations=activations_layer,
                                           drop_out = drop_out_layer,
                                           flag_evo=True)
                    model.fit(x_in_layer,x_in_layer, sample_weight=sample_weight_seq)
                    if flag_disp:
                        plt.plot(np.arange(0,n_epoch[0]+1), model.loss_)
                        plt.title('Autoencoder - Layer %d (%d neurons)'%(k+1,n_neu[k]))
                        plt.xlabel('Epochs')
                        plt.show()
                    
                    coefs_AE.append(model.coefs_[0])
                    x_in_layer = model.predict_hidden(x_in_layer)[0]
                
                # Training of the final AE layer (supervised)    
                if flag_disp:
                    logger.info('          x Training of Final Layer')
                    
                n_neu_layer=(n_neu[-1],)
                activations_layer=[activations[-2],activations[-1]]
                drop_out_layer = [drop_out[k],drop_out[k+1]]
                model = MLPAutoencoder(layers_size=n_neu_layer, n_epoch=n_epoch[0], n_batch=n_batch,
                                        noise_type=noise_type, noise_param=noise_param, noise_gen=noise_gen,                                       
                                        activations=activations_layer,
                                        drop_out = drop_out_layer,
                                        flag_evo=True)
                model.fit(x_in_layer,x_out, sample_weight=sample_weight)
                if flag_disp:
                    plt.plot(np.arange(0,n_epoch[0]+1), model.loss_)
                    plt.title('Autoencoder - Final Layer (%d neurons)'%(n_neu[-1]))
                    plt.xlabel('Epochs')
                    plt.show()
                 
                coefs_AE.append(model.coefs_[0])
                coefs_AE.append(model.coefs_[1])
                
                # Fine tuning of the global AE
                if flag_disp:
                    logger.info('          x Training of Full AE : Fine Tuning')
                
                model = MLPAutoencoder(layers_size=n_neu, n_epoch=n_epoch[-1], n_batch=n_batch,
                                        noise_type=noise_type, noise_param=noise_param, noise_gen=noise_gen,
                                        activations=activations,
                                        drop_out = drop_out,
                                        warm_start = True,
                                        flag_evo = True)
                
                model.coefs_ = coefs_AE
                
                model.fit(x_train,x_out, sample_weight=sample_weight)
                res_model.append(model.coefs_)
                if flag_disp:                    
                    plt.plot(np.arange(0,n_epoch[-1]+1), model.loss_)
                    plt.title('Autoencoder - Fine Tuning')
                    plt.xlabel('Epochs')
                    plt.show()
                                        
            else:
                # Global training of the AE as a supervised machine
                if type(n_epoch) == list:
                    n_epochs = n_epoch[-1]
                else:
                    n_epochs = n_epoch
                
                model = MLPAutoencoder(layers_size=n_neu, n_epoch=n_epochs, n_batch=n_batch,
                                       noise_type=noise_type, noise_param=noise_param, noise_gen=noise_gen,
                                       activations=activations,
                                       drop_out = drop_out,
                                       flag_evo=True)
                
                model.fit(x_train,x_out, sample_weight=sample_weight)
                if flag_disp:
                    plt.plot(np.arange(0,n_epochs+1), model.loss_)
                    plt.title('Autoencoder %s'%(str(n_neu)))
                    plt.xlabel('Epochs')
                    plt.show()
                    
                res_model.append(model.coefs_)
            
            # Evaluation of the trained solution
            o_train = model.predict(x_train)            
            o_test = model.predict(x_test)
            
            #if multitask_labels:
            ye_train_mt = np.zeros((y_train.shape[0],len(multitask_labels)+1))
            ye_test_mt = np.zeros((y_test.shape[0],len(multitask_labels)+1))
            
            for count in range(len(multitask_labels)+1):
                ye_train_mt[:,count] = (o_train[:,count]>threshold_dec).astype(int)
                ye_test_mt[:,count] = (o_test[:,count]>threshold_dec).astype(int)
                
                logger.info('            + BAcc (train) = %.4f BAcc (test) = %.4f'%(balanced_accuracy_score(y_train_mt[:,count],ye_train_mt[:,count]), balanced_accuracy_score(y_test_mt[:,count],ye_test_mt[:,count])))
                # Storing the decisions of the learner for each task
                ye_test_fold[kLearn,:,count] = ye_test_mt[:,count]                
        
        # The decisions of the different learners are averaged
        ye_test = np.median(ye_test_fold, axis=0).astype(int)
        # Plotting results and storing of the averaged decisions for the test set per fold
       #if multitask_labels:
        for kout in range(ye_test.shape[1]):
            TN, FP, FN, TP = confusion_matrix(y_test_mt[:,kout], ye_test[:,kout]).ravel()    
            logger.info(' ')
            logger.info(' Fold {}/{} : Acc = {}, Bal. Acc = {}'.format(kFold+1, num_folds, accuracy_score(y_test_mt[:,kout], ye_test[:,kout]), balanced_accuracy_score(y_test_mt[:,kout], ye_test[:,kout])))
            logger.info(' Fold {}/{} : TN = {}, FP = {}, FN = {}, TP = {} '.format(kFold+1, num_folds, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
                                        
        y_test_tot[idx_test_list[kFold],:] = y_test_mt
        ye_test_tot[idx_test_list[kFold],:] = ye_test
        
    # Processing of the obtained results
    col_label = 'Label'
    col_predict = 'Predict'
    col_result = 'Result'
    for kout in range(1+len(multitask_labels)):
        val_pred = ye_test_tot[:,kout] == 1
        df_alarms.insert(loc=df_alarms.shape[1], column=col_predict, value = val_pred)
        
        value_res =  df_alarms[col_label].astype(int).astype(str)+'-'+df_alarms[col_predict].astype(int).astype(str)    
        df_alarms.insert(loc=df_alarms.shape[1], column=col_result, value=value_res)
        df_alarms.replace({col_result: {'0-1': 'False Alarm', '1-0':'No Detection', '0-0': 'Accurate', '1-1': 'Accurate'}},inplace=True)
        
        col_label = 'Label%d'%(kout+1)
        col_predict = 'Predict%d'%(kout+1)
        col_result = 'Result%d'%(kout+1)    
    
    if num_folds == 1:
        df_alarms = df_alarms.loc[idx_test_list[0]]                                           
        
    if out_name != None:
        df_alarms.to_csv('{}_{}.csv'.format(out_name, model_name), index=False)
        logger.info('Predictions saved to {}'.format('{}_{}.csv'.format(out_name, model_name)))
    
    if out_model_path != None:
        with open(out_model_path, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(res_model, f)
        logger.info("Model saved to: {}".format(out_model_path))
                          
    return res_model, df_alarms, [balanced_accuracy_score(y_test_mt[:,kout], ye_test[:,kout]) for kout in range(ye_test.shape[1])]

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_predict_nfold_AE_Bayes(coded_data, res_models_AE, n_neu = (250,), n_epoch=100, n_batch=150, 
                       multitask_labels=None,
                       num_learners=11, use_NLP=True, 
                       activations=['tanh','identity'],
                       binary_type = 'PlusMinusOne',
                       dict_weight={0:1}, 
                       opt_weight=None,
                       drop_out = [],
                       class_cost={0:[1, 1]},
                       class_prob = {0:[0.5, 0.5]},
                       out_path = None,
                       flag_evo = True,
                       labels = ['False', 'True'],out_name=None, flag_disp=False):
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::                     
    #--------------------------------------------------------------------------
    # Organizing coded data
    #--------------------------------------------------------------------------        
    x_train_list, x_test_list, y_train_list, y_test_list, idx_test_list, coding_labels, nlp_feats, df_train_list, df_test_list, binary_type, normalization, text_enc= coded_data
   
    if isinstance(x_train_list, list):
        num_folds = len(x_train_list)
    else:   
        # For compatibility, the train and test data are converted in lists
        num_folds = 1
        x_train_list = [x_train_list]
        x_test_list = [x_test_list]
        y_train_list = [y_train_list]
        y_test_list = [y_test_list]
        df_train_list = [df_train_list]
        df_test_list = [df_test_list]
    
    df_alarms = pd.concat([df_train_list[0], df_test_list[0]]).sort_index()                    
    #--------------------------------------------------------------------------
    # Model training and evaluation
    #--------------------------------------------------------------------------
    model_name='Bayes MLP from AutoEncoder'
    res_model = []
    logger.info("Calculating results for Bayes MLP from AutoEncoder : {} neurons".format(n_neu))
    
    # Initialization of the total test labels and decision
    if not multitask_labels:
        multitask_labels = []
    
    y_test_tot = np.zeros((df_alarms.shape[0],len(multitask_labels)+1))
    ye_test_tot = np.zeros((df_alarms.shape[0],len(multitask_labels)+1))
    
    # Initiating the process per fold        
    for kFold in range(num_folds):        
        
        logger.info("  * Processing Fold {}".format(kFold+1))
        # Loading train and test data for the fold
        x_train = x_train_list[kFold]
        x_test = x_test_list[kFold] 
        y_train = y_train_list[kFold]
        y_test = y_test_list[kFold]
        # If NLP is False the text embeddings are removed from training data                                                                
        if use_NLP == False:
            x_train = x_train[:,0:len(coding_labels[kFold])]
            x_test = x_test[:,0:len(coding_labels[kFold])]            
                           
        # Generation of the sample weights and multitask labels                
        idx_train = list(set(range(df_alarms.shape[0])) - set(idx_test_list[kFold]))
            
        # Generation of multask train and test labels
        y_train_mt = np.zeros((y_train.shape[0],len(multitask_labels)+1))
        y_train_mt[:,0] = y_train
        y_test_mt = np.zeros((y_test.shape[0],len(multitask_labels)+1))
        y_test_mt[:,0] = y_test
        
        for count, label in enumerate(multitask_labels):
            if label == 'escalada':
                df_alarms['Label%d'%(count+1)] = (df_alarms['escalada_L2'] + df_alarms['escalada_procedimiento'])
            elif label in ['escalada_L2', 'escalada_procedimiento', 'reportado']:
                df_alarms['Label%d'%(count+1)] = df_alarms[label]
            
            y_train_mt[:,count+1] = df_alarms['Label%d'%(count+1)].loc[idx_train].astype(int).to_numpy()
            y_test_mt[:,count+1] = df_alarms['Label%d'%(count+1)].loc[idx_test_list[kFold]].astype(int).to_numpy()
            
        if opt_weight == 'operator':
            weight_operator = df_alarms['weight'].loc[idx_train].to_numpy()
            sample_weight = np.tile(weight_operator,(len(multitask_labels)+1,1)).T
        else:
            sample_weight = np.ones_like(y_train_mt)            
        
        for key in dict_weight.keys():
            sample_weight[:,key] = dict_weight[key] * sample_weight[:,key]
                
        ye_test_fold = np.zeros((num_learners,y_test.shape[0],len(multitask_labels)+1))            
                                                    
        logger.info("    + Size of training data : {}".format(x_train.shape))
        logger.info("    + Size of test data     : {}".format(x_test.shape))                        
        # Training of the AE learners
        for kLearn in range(num_learners):            
            logger.info('        * Training classifier - Fold {}/{} : Learner {} of {}'.format(kFold+1, num_folds, kLearn+1, num_learners))                
            
            # Global training of the AE as a supervised machine                        
            model = MLPBayesBinMT(layers_size=n_neu, n_epoch=n_epoch, n_batch=n_batch,                                   
                                   activations=activations,
                                   drop_out = drop_out,
                                   class_prob = class_prob,
                                   class_cost = class_cost,
                                   warm_start=True,
                                   flag_evo=flag_evo)
            
            coefs_AE = res_models_AE[kFold*num_folds + kLearn]
            coefs_AE[-1] = coefs_AE[-1][0:len(multitask_labels)+1]

            model.coefs_ = coefs_AE
            
            model.fit(x_train,y_train_mt, sample_weight=sample_weight)
            if flag_disp:
                plt.plot(np.arange(0,n_epoch+1), model.loss_)
                plt.title('MLP Bayes from Autoencoder %s'%(str(n_neu)))
                plt.xlabel('Epochs')
                plt.show()
                
            res_model.append(model.coefs_)
                        
            if multitask_labels:
                ye_train_mt = model.predict(x_train)
                ye_test_mt = model.predict(x_test)
                for count in range(len(multitask_labels)+1):
                    logger.info('          + BAcc (train) = %.3f BAcc (test) = %.3f'%(balanced_accuracy_score(y_train_mt[:,count],ye_train_mt[:,count]), balanced_accuracy_score(y_test_mt[:,count],ye_test_mt[:,count])))
                
                ye_test_fold[kLearn,:,:] = ye_test_mt
                
            else:
                ye_train = model.predict(x_train)
                ye_test = model.predict(x_test)
                logger.info('          + BAcc (train) = %.3f BAcc (test) = %.3f'%(balanced_accuracy_score(y_train,ye_train), balanced_accuracy_score(y_test,ye_test)))                                    
                # Storing the decisions of the learner (single task)        
                ye_test_fold[kLearn,:] = ye_test
        
        # The decisions of the different learners are averaged
        ye_test = np.median(ye_test_fold, axis=0).astype(int)
        # Plotting results and storing of the averaged decisions for the test set per fold
        for kout in range(ye_test.shape[1]):
            TN, FP, FN, TP = confusion_matrix(y_test_mt[:,kout], ye_test[:,kout]).ravel()    
            logger.info(' ')
            logger.info(' Fold {}/{} : Acc = {}, Bal. Acc = {}'.format(kFold+1, num_folds, accuracy_score(y_test_mt[:,kout], ye_test[:,kout]), balanced_accuracy_score(y_test_mt[:,kout], ye_test[:,kout])))
            logger.info(' Fold {}/{} : TN = {}, FP = {}, FN = {}, TP = {} '.format(kFold+1, num_folds, TN, FP, FN, TP))
            logger.info('   pFA = {} pM = {}'.format(FP/(TN+FP), FN/(TP+FN)))
            logger.info(' ')
                                        
        y_test_tot[idx_test_list[kFold],:] = y_test_mt
        ye_test_tot[idx_test_list[kFold],:] = ye_test
        
    # Processing of the obtained results
    col_label = 'Label'
    col_predict = 'Predict'
    col_result = 'Result'
    for kout in range(1+len(multitask_labels)):
        val_pred = ye_test_tot[:,kout] == 1
        df_alarms.insert(loc=df_alarms.shape[1], column=col_predict, value = val_pred)
        
        value_res =  df_alarms[col_label].astype(int).astype(str)+'-'+df_alarms[col_predict].astype(int).astype(str)    
        df_alarms.insert(loc=df_alarms.shape[1], column=col_result, value=value_res)
        df_alarms.replace({col_result: {'0-1': 'False Alarm', '1-0':'No Detection', '0-0': 'Accurate', '1-1': 'Accurate'}},inplace=True)
        
        col_label = 'Label%d'%(kout+1)
        col_predict = 'Predict%d'%(kout+1)
        col_result = 'Result%d'%(kout+1)    
    
    if num_folds == 1:
        df_alarms = df_alarms.loc[idx_test_list[0]]                                           
        
    if out_name != None:
        df_alarms.to_csv('{}_{}.csv'.format(out_name, model_name), index=False)
        logger.info('Predictions saved to {}'.format('{}_{}.csv'.format(out_name, model_name)))
    
    if out_path != None:
        with open(out_path, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(res_model, f)
        logger.info("Model saved to: {}".format(out_path))
                          
    return res_model, df_alarms

        