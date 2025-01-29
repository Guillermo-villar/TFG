#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:12 2023

@author: mlazaro
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries 
#import sys
import pandas as pd
import numpy as np
import logging
#import yaml
import os
#import json
#import ast
#from libraries.utils import convert_int

# Encoding

from tqdm import tqdm

import tensorflow_hub as hub
import tensorflow_text as text

from time import time
#from glob import glob
import gensim

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from nltk.tokenize import ToktokTokenizer

from sklearn.feature_extraction.text import CountVectorizer


# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

def get_labels_ext_fast(df_AL, df_INCs, inc_fw_win_size=[360,720,1440], inc_bw_win_size=15,
                        exclude_tagged_alarm=True, 
                        alarms_timestamp_colname='RootTime',
                        alarms_id_colname = 'ID',
                        alarms_site_colname = 'SiteCodInfo',
                        incs_timestamp_colname='Fecha-Hora Afectación',
                        incs_id_colname = 'I. Incident ID',
                        incs_site_colname = 'I. Centro Afectado'
                        ):
    
    # LINKING alarms with incidences
    logger.info('Linking alarms with incidences')    
    list_INCs_window = []
    list_INCs = []
    list_ALs = []
    list_ALs_prev = []
    
    df_AL_tmp = df_AL.copy()    
    for window_size in inc_fw_win_size:
        logger.info('Processing window size {}'.format(window_size))
        if exclude_tagged_alarm:
            df_AL_tmp = df_AL.loc[[x for x in df_AL.index if x not in list_ALs_prev]]        
            
        for kINC in [x for x in df_INCs.index if x not in list_INCs]:
            # logger.info('Processing inc {} with window {}'.format(kINC, window_size))
            timeINC = df_INCs.loc[kINC][incs_timestamp_colname]
            siteINC = df_INCs.loc[kINC][incs_site_colname]
            servINC = df_INCs.loc[kINC]['ServicesInc']
            df_AI = df_AL_tmp[(df_AL_tmp[alarms_timestamp_colname]>=(timeINC-pd.Timedelta(minutes=window_size))) 
                          & (df_AL_tmp[alarms_timestamp_colname]<=(timeINC+pd.Timedelta(minutes=inc_bw_win_size)))
                          & (df_AL_tmp[alarms_site_colname] == siteINC)]
            
            list_linked = []
            for nameSERV in servINC:
                aux = list(df_AI[df_AI['Services'].map(str).str.contains(nameSERV)].index)
                #if len(aux) > 0:
                #    list_linked += aux
                list_linked = list_linked + [x for x in aux if x not in list_linked]
                
            #list_linked  = list(set(list_linked))
            list_linked.sort()
            
            if len(list_linked)>0:
                list_INCs.append(kINC)
                list_INCs_window.append(window_size)
                list_ALs.append(list_linked)

        
        list_ALs_prev = []            
        for list_aux in list_ALs:
            list_ALs_prev += list_aux
            
    
    logger.info('Generating association dataframe')
    df_Anota = pd.DataFrame()
    auxWindow = []
    for k in range(len(list_INCs)):
        kINC = list_INCs[k]
        auxAL = df_AL.loc[list_ALs[k]].sort_values(by='ID')
        auxINC = df_INCs.loc[[kINC for x in range(auxAL.shape[0])]]
        auxWindow += [list_INCs_window[k] for x in range(auxAL.shape[0])] 
        aux = pd.concat([auxINC.reset_index(drop=True),auxAL.reset_index(drop=True)],axis=1)
        df_Anota = pd.concat([df_Anota,aux], axis=0, ignore_index=True)
    
    df_Anota.insert(loc = df_Anota.shape[1], column='WindowSizeMin', value=auxWindow)
    
    dict_Anota ={'timestamp_a': alarms_timestamp_colname, 
                  'timestamp_i': incs_timestamp_colname,
                  'id_a': alarms_id_colname,
                  'id_i': incs_id_colname,
                  'elem_name_a': 'SIRA',
                  'serv_name_a': 'Services',
                  'serv_name_i': 'ServicesInc',
    #              'type_a': ,
    #              'else_id',
                  'site_a': alarms_site_colname,
                  'site_i': incs_site_colname,
    #              'impact_i',
    #              'urgency_i',
    #              'priority_i',
    #              'ioc_t1',
    #              'causa_i',                 
    #              'severity_a', 
    #              'client_i', 
                  'summary_i': 'I. Summary',
                  'window_size': 'WindowSizeMin'
        }
    
    df_Anota_Simple = pd.DataFrame()
    kloc = 0
    for key in dict_Anota:
        #print(dict_Anota[key])        
        df_Anota_Simple.insert(loc=kloc, column=key, value=df_Anota[dict_Anota[key]])
        kloc += 1
        
    return df_Anota, df_Anota_Simple

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

def coding_BIN(df_train, df_test, binary_feats, binary_type):
    
    if len(binary_feats)>0:
        BIN_train=pd.DataFrame(columns=binary_feats)
        BIN_test=pd.DataFrame(columns=binary_feats)
        
        for kbin in range(len(binary_feats)):
            aux_train = pd.get_dummies(data=df_train[binary_feats[kbin]], columns=binary_feats[kbin])
            aux_test = pd.get_dummies(data=df_test[binary_feats[kbin]], columns=binary_feats[kbin])
            aux_train, aux_test = aux_train.align(aux_test, join='left', axis=1)
            
            if binary_type == 'OneZero':
                BIN_train[binary_feats[kbin]] = aux_train[aux_train.columns[0]] # - aux_train[aux_train.columns[1]]
                BIN_test[binary_feats[kbin]] = aux_test[aux_test.columns[0]] # - aux_test[aux_test.columns[1]]
            elif binary_type == 'PlusMinusOne':
                BIN_train[binary_feats[kbin]] = aux_train[aux_train.columns[0]] - aux_train[aux_train.columns[1]]
                BIN_test[binary_feats[kbin]] = aux_test[aux_test.columns[0]] - aux_test[aux_test.columns[1]]
                        
        BIN_labels = list(BIN_train.columns) 
        BIN_train = BIN_train.to_numpy(dtype=np.float64)
        BIN_test = BIN_test.to_numpy(dtype=np.float64)
            
    else:
        BIN_train=np.zeros((df_train.shape[0],0))
        BIN_test=np.zeros((df_test.shape[0],0))
        BIN_labels = []
    
    
    return BIN_train, BIN_test, BIN_labels
    
def coding_NUM(df_train, df_test, numeric_feats, normalization):    
    if len(numeric_feats)>0:
        if normalization == 'MinMax':
            scaler = MinMaxScaler()
        elif normalization == 'MinMaxOne':
            scaler = MinMaxScaler(feature_range=(-1,1))
        elif normalization == 'MaxAbs':
            scaler = MaxAbsScaler()    
        elif normalization == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = StandardScaler()
            
        NUM_labels = numeric_feats
        NUM_train = scaler.fit_transform(df_train[numeric_feats].fillna(0).to_numpy(dtype=np.float64))
        NUM_test = scaler.transform(df_test[numeric_feats].fillna(0).to_numpy(dtype=np.float64))        
    else:
        NUM_train=np.zeros((df_train.shape[0],0))
        NUM_test=np.zeros((df_test.shape[0],0))
        NUM_labels = []
        
    return NUM_train, NUM_test, NUM_labels
    

def coding_OHE(df_train, df_test, purely_categorical_feats):
    
    if len(purely_categorical_feats) >0:
        OHE_train = pd.get_dummies(data=df_train[purely_categorical_feats], columns=purely_categorical_feats)
        OHE_test = pd.get_dummies(data=df_test[purely_categorical_feats], columns=purely_categorical_feats)
    
        # OHE values are aligned with the training set
        OHE_train, OHE_test = OHE_train.align(OHE_test, join='left', axis=1)
    
        OHE_test.fillna(0, inplace=True)
        
        OHE_labels = list(OHE_train.columns)
        OHE_train = OHE_train.to_numpy(dtype=np.float64)
        OHE_test = OHE_test.to_numpy(dtype=np.float64)
    else:
        OHE_train=np.zeros((df_train.shape[0],0))
        OHE_test=np.zeros((df_test.shape[0],0))
        OHE_labels=[]
        
    return OHE_train, OHE_test, OHE_labels


def coding_MHE(df_train, df_test, multiple_categorical_feats):
    
    if len(multiple_categorical_feats) >0:
        cats2remove=['','missing']
        MHE_train = np.array([], dtype=np.float64).reshape(df_train.shape[0],0)
        MHE_test = np.array([], dtype=np.float64).reshape(df_test.shape[0],0)
        MHE_labels=[]
        
        for feat in multiple_categorical_feats:
            feat_values_train = df_train[feat].fillna('').apply(str).apply(lambda x: [s for s in x.lower().split(', ')])
            feat_values_test = df_test[feat].fillna('').apply(str).apply(lambda x: [s for s in x.lower().split(', ')])
            
            feat_categories = []
            for elem in feat_values_train.value_counts().index.to_list():
                feat_categories += elem
                
            feat_categories = list(set(feat_categories)) 
            feat_categories.sort()
            feat_categories = [x for x in feat_categories if x not in cats2remove]
            
            feat_df_train = pd.DataFrame()
            feat_df_test = pd.DataFrame()
            for cat in feat_categories:
                #feat_df_train.insert(loc=feat_df_train.shape[1], column=feat+'_'+cat, 
                #               value=feat_values_train.apply(lambda x: cat in x))
                #feat_df_test.insert(loc=feat_df_test.shape[1], column=feat+'_'+cat, 
                #               value=feat_values_test.apply(lambda x: cat in x))
                
                # Modified to avoid Perfomance Warning (DataFrame is highly fragmented)
                feat_df_train = pd.concat([feat_df_train,
                                           pd.DataFrame({feat+'_'+cat: feat_values_train.apply(lambda x: cat in x)})],axis=1)
                feat_df_test = pd.concat([feat_df_test,
                                           pd.DataFrame({feat+'_'+cat: feat_values_test.apply(lambda x: cat in x)})],axis=1)
                
            
            MHE_labels += list(feat_df_train.columns)
            MHE_train = np.hstack([MHE_train,feat_df_train.astype(int).to_numpy()])
            MHE_test = np.hstack([MHE_test,feat_df_test.astype(int).to_numpy()])
                    
        
    else:
        MHE_train=np.zeros((df_train.shape[0],0))
        MHE_test=np.zeros((df_test.shape[0],0))
        MHE_labels=[]
        
    return MHE_train, MHE_test, MHE_labels
    
def coding_NLP(df_train_NLP, df_test_NLP, text_enc, model_path):
    
    ### Generación de Características (Count Vectorizer)
    if text_enc == 'CountVectorizer':        
         
        logger.info('   %s : Processing column (Train)'%(text_enc))                                               
        auxList = df_train_NLP.fillna('empty').tolist()
        
        
        vectorizer = CountVectorizer(min_df=1)#, max_features=50)
        vectors_SKL = vectorizer.fit_transform(auxList)  # Sparse matrix
        vectors_SKL_mat = vectors_SKL.todense()
        
        NLP_train = vectors_SKL_mat
            
        logger.info('   %s : Processing column (Test)'%(text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna('empty').tolist()
            vectors_SKL = vectorizer.transform(auxList)  # Sparse matrix
            vectors_SKL_mat = vectors_SKL.todense()
        
            NLP_test = vectors_SKL_mat
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0],0))
                  

    ### Generación de Características (FastText, Gensim)
    elif 'FastText' in text_enc:    
        # FastTest works faster if size is multiple of 4
        size_FT = int(text_enc[8:])
                                                    
        tokenizer = ToktokTokenizer()
                
        logger.info('   FastText : Processing (Train)')     
            
        auxList = df_train_NLP.fillna('empty').tolist()
        auxToken  = [tokenizer.tokenize(doc) for doc in auxList]
                
        model_FT = gensim.models.FastText(min_count=1, vector_size=size_FT, workers=1)
        model_FT.build_vocab(auxToken, progress_per=1000)
            
        t = time()
        model_FT.train(auxToken, total_examples=model_FT.corpus_count, epochs=30, report_delay=1)         
        logger.info('Training time (FastText model): {} minutes'.format(round((time() - t) / 60, 2)))
        logger.info(model_FT)
            
        #words_FT = list(set(x for l in auxToken for x in l))   
        vectors_FT_tot = np.array([np.array([model_FT.wv[i] for i in ls]) for ls in auxToken])
        vectors_FT_mat = []
        for v in vectors_FT_tot:
            if v.size:
                vectors_FT_mat.append(v.mean(axis=0))
            else:
                vectors_FT_mat.append(np.zeros(size_FT, dtype=float))
            
        NLP_train = np.array(vectors_FT_mat)
        
        logger.info('   %s : Processing column (Test)'%(text_enc))
        if df_test_NLP.shape[0] > 0:                        
            auxList = df_test_NLP.fillna('empty').tolist()
            auxToken  = [tokenizer.tokenize(doc) for doc in auxList]
            vectors_FT_tot = np.array([np.array([model_FT.wv[i] for i in ls]) for ls in auxToken])
            vectors_FT_mat = []
            for v in vectors_FT_tot:
                if v.size:
                    vectors_FT_mat.append(v.mean(axis=0))
                else:
                    vectors_FT_mat.append(np.zeros(size_FT, dtype=float))
                            
            NLP_test = np.array(vectors_FT_mat)
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0],0))
                        

    ### Generación de características (CBOW, Word2Vec)
    elif 'Word2Vec' in text_enc:
        # Works faster if size is multiple of 4
        size_WV = int(text_enc[8:])
                        
        tokenizer = ToktokTokenizer() 
            
        logger.info('   %s : Processing column (Train)'%(text_enc))
                  
        auxList = df_train_NLP.fillna('empty').tolist()
        auxToken  = [tokenizer.tokenize(doc) for doc in auxList]
        
        model_WV = gensim.models.Word2Vec(min_count=1, vector_size=size_WV, workers=1)
        model_WV.build_vocab(auxToken, progress_per=1000)
        
        t = time()
        model_WV.train(auxToken, total_examples=model_WV.corpus_count, epochs=300, report_delay=1)         
        logger.info('  Training time (Word2Vec model): {} minutes'.format(round((time() - t) / 60, 2)))
        # Se cierra el aprendizaje del modelo para salvar memoria 
        #model_WV.init_sims(replace=True)        
        logger.info(model_WV)
        
        words_WV = list(set(x for l in auxToken for x in l))
        vectors_WV_tot = np.array([np.array([model_WV.wv[i] for i in ls]) for ls in auxToken])
        vectors_WV_mat = []
        for v in vectors_WV_tot:
            if v.size:
                vectors_WV_mat.append(v.mean(axis=0))
            else:
                vectors_WV_mat.append(np.zeros(size_WV, dtype=float))
        
        NLP_train = np.array(vectors_WV_mat)
                
        logger.info('   %s : Processing column (Test)'%(text_enc))
        if df_test_NLP.shape[0] > 0:                
            auxList = df_test_NLP.fillna('empty').tolist()
            auxToken  = [tokenizer.tokenize(doc) for doc in auxList]
            vectors_WV_tot = np.array([np.array([model_WV.wv[i] for i in ls if all(item in words_WV for item in ls)]) for ls in auxToken])
            vectors_WV_mat = []
            for v in vectors_WV_tot:
                if v.size:
                    vectors_WV_mat.append(v.mean(axis=0))
                else:
                    vectors_WV_mat.append(np.zeros(size_WV, dtype=float))
            
            NLP_test = np.array(vectors_WV_mat)
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0],0))
        

    elif text_enc in ['GoogleNews20', 'GoogleNews128', 'NNLM128', 'CNNmulti512']:            
        NLP_train = np.array([], dtype=np.float64).reshape(df_train_NLP.shape[0],0)
        NLP_test = np.array([], dtype=np.float64).reshape(df_test_NLP.shape[0],0)
        
        batch_size=4096
        
        if text_enc == 'GoogleNews20':
            embed = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")
        elif text_enc == 'GoogleNews128':
            embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
        elif text_enc == 'NNLM128':
            #embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1")
            embed = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")
        elif text_enc == 'CNNmulti512':
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
                
        auxList = df_train_NLP.fillna('empty').tolist()
        
        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):
            print('   EMBED {}'.format(ii))
            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList)-ii
                break_flag = True

            batch = auxList[ii:ii+batch_size]
            if ii == 0:
                vectors_NLP = embed(batch)
            else:
                vectors_NLP = np.append(vectors_NLP,
                                    embed(batch),
                                    axis=0)
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train,vectors_NLP])
    
        logger.info('   %s : Processing column (Test)'%(text_enc))
        
        if df_test_NLP.shape[0] > 0:
                        
            auxList = df_test_NLP.fillna('empty').tolist()                
            
            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):

                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList)-ii
                    break_flag = True

                batch = auxList[ii:ii+batch_size]
                if ii == 0:
                    vectors_NLP = embed(batch)
                else:
                    vectors_NLP = np.append(vectors_NLP,
                                        embed(batch),
                                        axis=0)
                if break_flag:
                    break

            NLP_test = np.hstack([NLP_test,vectors_NLP])
                            
    elif text_enc in ['BERT128','LaBSE']:
        NLP_train = np.array([], dtype=np.float64).reshape(df_train_NLP.shape[0],0)
        NLP_test = np.array([], dtype=np.float64).reshape(df_test_NLP.shape[0],0)
        
        if text_enc == 'BERT128':
            batch_size=128
            preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
            encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2",
                                     trainable=True)
        elif text_enc == 'LaBSE':
            batch_size=256
            #preprocessor = hub.KerasLayer(
            #  "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2") # hub.KerasLayer(os.path.join(model_path, 'LaBSE_v2_preprocessor'))
            #encoder =  hub.KerasLayer("https://tfhub.dev/google/LaBSE/2") # hub.KerasLayer(os.path.join(model_path, 'LaBSE_v2'))
            preprocessor = hub.KerasLayer(os.path.join(model_path, 'LaBSE_v2_preprocessor'))
            encoder =  hub.KerasLayer(os.path.join(model_path, 'LaBSE_v2'))
                                            
    
        logger.info('   %s : Processing column (Train)'%(text_enc))
        
        auxList = df_train_NLP.fillna('empty').tolist()        
        
        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):

            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList)-ii
                break_flag = True

            batch = auxList[ii:ii+batch_size]
            if ii == 0:
                vectors_NLP = encoder(preprocessor(batch))['default'].numpy()
            else:
                vectors_NLP = np.append(vectors_NLP,
                                    encoder(preprocessor(batch))['default'].numpy(),
                                    axis=0)
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train,vectors_NLP])
                            
        logger.info('   %s : Processing column (Test)'%(text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna('empty').tolist()
            
            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):
    
                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList)-ii
                    break_flag = True
    
                batch = auxList[ii:ii+batch_size]
                if ii == 0:
                    vectors_NLP = encoder(preprocessor(batch))['default'].numpy()
                else:
                    vectors_NLP = np.append(vectors_NLP,
                                        encoder(preprocessor(batch))['default'].numpy(),
                                        axis=0)
                if break_flag:
                    break
    
            NLP_test = np.hstack([NLP_test,vectors_NLP])
            
    else:
        print('Text encodign not implemented')
    
    
    return NLP_train, NLP_test

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def coding_all_columns(df_train, df_test,
                       binary_feats,
                       binary_type,
                       numeric_feats,
                       normalization,
                       purely_categorical_feats,
                       multiple_categorical_feats,
                       nlp_feats,
                       text_enc,
                       model_path,
        ):
    """
    Encoding of data with text embeddings common for all text columns

    Parameters
    ----------
    df_train : dataframe
        Dataframe with train data
    df_test : dataframe
        Dataframe with test data
    binary_feats : list of str
        List with binary features in df_train and df_test
    binary_type : str
        Type of binary encoding ('OneZero', 'PlusMinusOne')
    numeric_feats : list of str
        List with numeric features in df_train and df_test
    normalization : str
        Type of normalization used for numeric features ('Standard', 'MinMax', 'MinMaxOne', 'MaxAbs')
    purely_categorical_feats : list of ste
        List with purely categorical features in df_train and df_test
    multiple_categorical_feats : list of str
        List with multi-categorical features in df_train and df_test
    nlp_feats : list of str
        List with text features in df_train and df_test (encoded using NLP methods)
    text_enc : str
        Type of NLP embedding
    model_path : str
        Path to the model
     : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train: numpy array
        Array with coded train data
    y_train:  numpy array
        Array with binary train labels
    x_test: numpy array
        Array with coded test data
    y_test: numpy array
        Array with binary test data
    coding_labels: list of str
        List with the names of the coding labels for every feature (except the NLP features)

    """
    
    # Numeric features
    logger.info("  *  Encoding numeric features: {}".format(numeric_feats))
    NUM_train, NUM_test, NUM_labels = coding_NUM(df_train, df_test, numeric_feats, normalization)

    # Binary features
    logger.info("  *  Encoding binary features: {}".format(binary_feats))
    BIN_train, BIN_test, BIN_labels = coding_BIN(df_train, df_test, binary_feats, binary_type)
        
    
    # Binary data
    if len(binary_feats)>0:
        BIN_train=pd.DataFrame(columns=binary_feats)
        BIN_test=pd.DataFrame(columns=binary_feats)
        
        for kbin in range(len(binary_feats)):
            aux_train = pd.get_dummies(data=df_train[binary_feats[kbin]], columns=binary_feats[kbin])
            aux_test = pd.get_dummies(data=df_test[binary_feats[kbin]], columns=binary_feats[kbin])
            aux_train, aux_test = aux_train.align(aux_test, join='left', axis=1)
            
            if binary_type == 'OneZero':
                BIN_train[binary_feats[kbin]] = aux_train[aux_train.columns[0]] # - aux_train[aux_train.columns[1]]
                BIN_test[binary_feats[kbin]] = aux_test[aux_test.columns[0]] # - aux_test[aux_test.columns[1]]
            elif binary_type == 'PlusMinusOne':
                BIN_train[binary_feats[kbin]] = aux_train[aux_train.columns[0]] - aux_train[aux_train.columns[1]]
                BIN_test[binary_feats[kbin]] = aux_test[aux_test.columns[0]] - aux_test[aux_test.columns[1]]
                        
        BIN_labels = list(BIN_train.columns) 
        BIN_train = BIN_train.to_numpy(dtype=np.float64)
        BIN_test = BIN_test.to_numpy(dtype=np.float64)
            
    else:
        BIN_train=np.zeros((df_train.shape[0],0))
        BIN_test=np.zeros((df_test.shape[0],0))
        BIN_labels = []
    # One-Hot-Encoding (OHE) of all the data sets 
    logger.info("  *  Encoding categorical features: {}".format(purely_categorical_feats))
    OHE_train, OHE_test, OHE_labels = coding_OHE(df_train,df_test,purely_categorical_feats)    
    
    # Multiple-Hot-Encoding (MHE)
    logger.info("  *  Encoding multi-categorical features: {}".format(multiple_categorical_feats))
    MHE_train, MHE_test, MHE_labels = coding_MHE(df_train,df_test,multiple_categorical_feats)
        
    # NLP features
    logger.info("  *  Encoding text (NLP) features: {}".format(nlp_feats))
    if len(nlp_feats) > 0:
        ### Combination of all text columns:
        logger.info('Combining %d columns (Train)...'%(len(nlp_feats)))     
        joint_text = df_train[nlp_feats].astype(str).agg(' '.join, axis=1)
        df_train.insert(loc=1, column='Joint Text', value=joint_text)
                
        logger.info('Combining %d columns (Test)...'%(len(nlp_feats)))
        if df_test.shape[0] > 0:
            joint_text = df_test[nlp_feats].astype(str).agg(' '.join, axis=1)
            df_test.insert(loc=1, column='Joint Text', value=joint_text)
            NLP_train, NLP_test = coding_NLP(df_train['Joint Text'], df_test['Joint Text'], text_enc, model_path)
            
        else:
            NLP_train, NLP_test = coding_NLP(df_train['Joint Text'], df_test, text_enc, model_path)
                                                                                                           
    else: 
        NLP_train=np.zeros((df_train.shape[0],0))
        NLP_test=np.zeros((df_test.shape[0],0))
          
                                         
    # Labels of encoded characteristics
    coding_labels = NUM_labels + BIN_labels + OHE_labels + MHE_labels
    
    # Train and test patterns and labels
    x_train = np.hstack([NUM_train, BIN_train, OHE_train, MHE_train, NLP_train])        
    y_train = df_train['Label'].replace({True: 1, False: 0}).to_numpy()
    if df_test.shape[0] > 0:
        x_test = np.hstack([NUM_test, BIN_test, OHE_test, MHE_test, NLP_test])
        y_test = df_test['Label'].replace({True: 1, False: 0}).to_numpy()
    else:
        x_test = 0
        y_test = 0
         
    return x_train, y_train, x_test, y_test, coding_labels

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def coding_per_columns(df_train, df_test,
                       binary_feats,
                       binary_type,
                       numeric_feats,
                       normalization,
                       purely_categorical_feats,
                       multiple_categorical_feats,
                       nlp_feats,
                       text_enc,
                       model_path,
        ):
    """
    Encoding of data with text embeddings per text column

    Parameters
    ----------
    df_train : dataframe
        Dataframe with train data
    df_test : dataframe
        Dataframe with test data
    binary_feats : list of str
        List with binary features in df_train and df_test
    binary_type : str
        Type of binary encoding ('OneZero', 'PlusMinusOne')
    numeric_feats : list of str
        List with numeric features in df_train and df_test
    normalization : str
        Type of normalization used for numeric features ('Standard', 'MinMax', 'MinMaxOne', 'MaxAbs')
    purely_categorical_feats : list of ste
        List with purely categorical features in df_train and df_test
    multiple_categorical_feats : list of str
        List with multi-categorical features in df_train and df_test
    nlp_feats : list of str
        List with text features in df_train and df_test (encoded using NLP methods)
    text_enc : str
        Type of NLP embedding
    model_path : str
        Path to the model
     : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train: numpy array
        Array with coded train data
    y_train:  numpy array
        Array with binary train labels
    x_test: numpy array
        Array with coded test data
    y_test: numpy array
        Array with binary test data
    coding_labels: list of str
        List with the names of the coding labels for every feature (except the NLP features)

    """
    
    # Numeric features
    logger.info("  *  Encoding numeric features: {}".format(numeric_feats))
    NUM_train, NUM_test, NUM_labels = coding_NUM(df_train, df_test, numeric_feats, normalization)

    # Binary features
    logger.info("  *  Encoding binary features: {}".format(binary_feats))
    BIN_train, BIN_test, BIN_labels = coding_BIN(df_train, df_test, binary_feats, binary_type)
        
    # One-Hot-Encoding (OHE) of all the data sets
    logger.info("  *  Encoding categorical features: {}".format(purely_categorical_feats))
    OHE_train, OHE_test, OHE_labels = coding_OHE(df_train,df_test,purely_categorical_feats)    
    
    # Multiple-Hot-Encoding (MHE)
    logger.info("  *  Encoding multi-categorical features: {}".format(multiple_categorical_feats))
    MHE_train, MHE_test, MHE_labels = coding_MHE(df_train,df_test,multiple_categorical_feats)
    
    
    # NLP features
    logger.info("  *  Encoding text (NLP) features: {}".format(nlp_feats))
    NLP_train=np.zeros((df_train.shape[0],0))
    NLP_test=np.zeros((df_test.shape[0],0))
    if len(nlp_feats) > 0:
        for feat in nlp_feats:
            aux_train, aux_test = coding_NLP(df_train[feat], df_test[feat], text_enc, model_path)
            
            NLP_train = np.hstack([NLP_train,aux_train])
            NLP_test = np.hstack([NLP_test,aux_test])
                                                                                                                                     
                                         
    # Labels of encoded characteristics
    coding_labels = NUM_labels + BIN_labels + OHE_labels + MHE_labels
    
    # Train and test patterns and labels
    x_train = np.hstack([NUM_train, BIN_train, OHE_train, MHE_train, NLP_train])        
    y_train = df_train['Label'].replace({True: 1, False: 0}).to_numpy()
    if df_test.shape[0] > 0:
        x_test = np.hstack([NUM_test, BIN_test, OHE_test, MHE_test, NLP_test])
        y_test = df_test['Label'].replace({True: 1, False: 0}).to_numpy()
    else:
        x_test = 0
        y_test = 0
        
    
    return x_train, y_train, x_test, y_test, coding_labels
