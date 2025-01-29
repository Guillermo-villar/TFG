#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 17:21:12 2023

@author: mlazaro
"""


import json
import logging
import os
import pickle
import random
from time import time
from typing import Dict

import gensim
import joblib
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# basic python libraries
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import yaml
from feature_engine.encoding import OneHotEncoder
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZE LOGGER
# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
    format=FORMAT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


def coding_BIN(df_train, df_test, binary_feats, binary_type):
    if len(binary_feats) > 0:
        # Initialize the DataFrames for binary encoded features
        BIN_train = pd.DataFrame(index=df_train.index)
        BIN_test = pd.DataFrame(index=df_test.index)

        for feature in binary_feats:
            # Get dummies for the feature in both train and test sets
            aux_train = pd.get_dummies(df_train[feature], drop_first=False)
            aux_test = pd.get_dummies(df_test[feature], drop_first=False)

            # Align the dummy columns in train and test sets
            aux_train, aux_test = aux_train.align(
                aux_test, join="left", axis=1, fill_value=0
            )

            # Determine the binary encoding
            if binary_type == "OneZero":
                BIN_train[feature] = aux_train.iloc[:, 0].astype(int)
                BIN_test[feature] = aux_test.iloc[:, 0].astype(int)
            elif binary_type == "PlusMinusOne":
                BIN_train[feature] = aux_train.iloc[:, 0].astype(
                    int
                ) - aux_train.iloc[:, 1].astype(int)
                BIN_test[feature] = aux_test.iloc[:, 0].astype(
                    int
                ) - aux_test.iloc[:, 1].astype(int)

        BIN_labels = list(BIN_train.columns)
        BIN_train = BIN_train.to_numpy(dtype=np.float64)
        BIN_test = BIN_test.to_numpy(dtype=np.float64)
    else:
        BIN_train = np.zeros((df_train.shape[0], 0))
        BIN_test = np.zeros((df_test.shape[0], 0))
        BIN_labels = []

    return BIN_train, BIN_test, BIN_labels


def coding_BIN2(
    df_train,
    df_test,
    binary_feats,
    bincat_dict,
    binary_type,
    subversion="prueba",
):
    if len(binary_feats) > 0:
        for key, value in bincat_dict.items():
            df_train[key] = df_train[key] == value
            df_test[key] = df_test[key] == value

        if binary_type == "OneZero":
            df_train[binary_feats] = df_train[binary_feats].astype(int)
            df_test[binary_feats] = df_test[binary_feats].astype(int)
        elif binary_type == "PlusMinusOne":
            df_train[binary_feats] = df_train[binary_feats].astype(int) * 2 - 1
            df_test[binary_feats] = df_test[binary_feats].astype(int) * 2 - 1

        BIN_labels = binary_feats
        BIN_train = df_train[binary_feats].to_numpy(dtype=np.float64)
        BIN_test = df_test[binary_feats].to_numpy(dtype=np.float64)

    else:
        BIN_train = np.zeros((df_train.shape[0], 0))
        BIN_test = np.zeros((df_test.shape[0], 0))
        BIN_labels = []

    with open("config/inference_config_" + subversion + ".yaml", "r") as file:
        yaml_dict = yaml.safe_load(file)

    yaml_dict["params"]["binary_feats"] = BIN_labels
    yaml_dict["params"]["bincat_dict"] = bincat_dict
    yaml_dict["params"]["binary_type"] = binary_type

    with open("config/inference_config_" + subversion + ".yaml", "w") as file:
        yaml.dump(yaml_dict, file, default_flow_style=False)

    return BIN_train, BIN_test, BIN_labels


def coding_NUM(df_train, df_test, numeric_feats, normalization):
    if len(numeric_feats) > 0:
        if normalization == "MinMax":
            scaler = MinMaxScaler()
        elif normalization == "MinMaxOne":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalization == "MaxAbs":
            scaler = MaxAbsScaler()
        elif normalization == "Standard":
            scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        NUM_labels = numeric_feats
        NUM_train = scaler.fit_transform(
            df_train[numeric_feats].fillna(0).to_numpy(dtype=np.float64)
        )
        if df_test.shape[0] > 0:
            NUM_test = scaler.transform(
                df_test[numeric_feats].fillna(0).to_numpy(dtype=np.float64)
            )
        else:
            NUM_test = np.zeros((df_test.shape[0], 0))

    else:
        NUM_train = np.zeros((df_train.shape[0], 0))
        NUM_test = np.zeros((df_test.shape[0], 0))
        NUM_labels = []

    return NUM_train, NUM_test, NUM_labels


def coding_NUM2(
    df_train, df_test, numeric_feats, normalization, subversion="prueba"
):
    if len(numeric_feats) > 0:
        if normalization == "MinMax":
            scaler = MinMaxScaler()
        elif normalization == "MinMaxOne":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalization == "MaxAbs":
            scaler = MaxAbsScaler()
        elif normalization == "Standard":
            scaler = StandardScaler()
        else:
            scaler = StandardScaler()

        NUM_labels = numeric_feats
        NUM_train = scaler.fit_transform(
            df_train[numeric_feats].fillna(0).to_numpy(dtype=np.float64)
        )
        if df_test.shape[0] > 0:
            NUM_test = scaler.transform(
                df_test[numeric_feats].fillna(0).to_numpy(dtype=np.float64)
            )
        else:
            NUM_test = np.zeros((df_test.shape[0], 0))

    else:
        NUM_train = np.zeros((df_train.shape[0], 0))
        NUM_test = np.zeros((df_test.shape[0], 0))
        NUM_labels = []

    with open("config/inference_config_" + subversion + ".yaml", "r") as file:
        yaml_dict = yaml.safe_load(file)

    joblib.dump(
        scaler,
        os.path.join(
            yaml_dict["deps"]["model"]["model_path"],
            "scaler_model_" + subversion + ".joblib",
        ),
    )

    yaml_dict["params"]["numeric_feats"] = NUM_labels

    with open("config/inference_config_" + subversion + ".yaml", "w") as file:
        yaml.dump(yaml_dict, file, default_flow_style=False)

    return NUM_train, NUM_test, NUM_labels


def coding_OHE(df_train, df_test, purely_categorical_feats):

    if len(purely_categorical_feats) > 0:
        OHE_train = pd.get_dummies(
            data=df_train[purely_categorical_feats],
            columns=purely_categorical_feats,
        )
        OHE_test = pd.get_dummies(
            data=df_test[purely_categorical_feats],
            columns=purely_categorical_feats,
        )

        # OHE values are aligned with the training set
        OHE_train, OHE_test = OHE_train.align(OHE_test, join="left", axis=1)
        OHE_test.fillna(0, inplace=True)

        OHE_labels = list(OHE_train.columns)
        OHE_train = OHE_train.to_numpy(dtype=np.float64)
        OHE_test = OHE_test.to_numpy(dtype=np.float64)
    else:
        OHE_train = np.zeros((df_train.shape[0], 0))
        OHE_test = np.zeros((df_test.shape[0], 0))
        OHE_labels = []

    return OHE_train, OHE_test, OHE_labels


def coding_OHE2(
    df_train, df_test, purely_categorical_feats, subversion="prueba"
):

    if len(purely_categorical_feats) > 0:
        ohe = OneHotEncoder()
        OHE_train = ohe.fit_transform(df_train[purely_categorical_feats])
        # Save Joblib
        OHE_test = ohe.transform(df_test[purely_categorical_feats])
        OHE_test.fillna(0, inplace=True)

        OHE_labels = list(OHE_train.columns)
        OHE_train = OHE_train.to_numpy(dtype=np.float64)
        OHE_test = OHE_test.to_numpy(dtype=np.float64)
    else:
        OHE_train = np.zeros((df_train.shape[0], 0))
        OHE_test = np.zeros((df_test.shape[0], 0))
        OHE_labels = []

    with open("config/inference_config_" + subversion + ".yaml", "r") as file:
        yaml_dict = yaml.safe_load(file)

    joblib.dump(
        ohe,
        os.path.join(
            yaml_dict["deps"]["model"]["model_path"],
            "ohe_model_" + subversion + ".joblib",
        ),
    )

    yaml_dict["params"]["purely_categorical_feats"] = purely_categorical_feats
    yaml_dict["params"]["ohe_feats"] = OHE_labels

    with open("config/inference_config_" + subversion + ".yaml", "w") as file:
        yaml.dump(yaml_dict, file, default_flow_style=False)

    return OHE_train, OHE_test, OHE_labels


def coding_MHE(df_train, df_test, multiple_categorical_feats):

    if len(multiple_categorical_feats) > 0:
        cats2remove = ["", "missing"]
        MHE_train = np.array([], dtype=np.float64).reshape(
            df_train.shape[0], 0
        )
        MHE_test = np.array([], dtype=np.float64).reshape(df_test.shape[0], 0)
        MHE_labels = []

        for feat in multiple_categorical_feats:
            feat_values_train = (
                df_train[feat]
                .fillna("")
                .apply(str)
                .apply(lambda x: [s for s in x.lower().split(", ")])
            )
            feat_values_test = (
                df_test[feat]
                .fillna("")
                .apply(str)
                .apply(lambda x: [s for s in x.lower().split(", ")])
            )

            feat_categories = []
            for elem in feat_values_train.value_counts().index.to_list():
                feat_categories += elem

            feat_categories = list(set(feat_categories))
            feat_categories.sort()
            feat_categories = [
                x for x in feat_categories if x not in cats2remove
            ]

            feat_df_train = pd.DataFrame()
            feat_df_test = pd.DataFrame()
            for cat in feat_categories:
                # feat_df_train.insert(loc=feat_df_train.shape[1], column=feat+"_"+cat,
                #               value=feat_values_train.apply(lambda x: cat in x))
                # feat_df_test.insert(loc=feat_df_test.shape[1], column=feat+"_"+cat,
                #               value=feat_values_test.apply(lambda x: cat in x))

                # Modified to avoid Perfomance Warning (DataFrame is highly fragmented)
                feat_df_train = pd.concat(
                    [
                        feat_df_train,
                        pd.DataFrame(
                            {
                                feat
                                + "_"
                                + cat: feat_values_train.apply(
                                    lambda x: cat in x
                                )
                            }
                        ),
                    ],
                    axis=1,
                )
                feat_df_test = pd.concat(
                    [
                        feat_df_test,
                        pd.DataFrame(
                            {
                                feat
                                + "_"
                                + cat: feat_values_test.apply(
                                    lambda x: cat in x
                                )
                            }
                        ),
                    ],
                    axis=1,
                )

            MHE_labels += list(feat_df_train.columns)
            MHE_train = np.hstack(
                [MHE_train, feat_df_train.astype(int).to_numpy()]
            )
            MHE_test = np.hstack(
                [MHE_test, feat_df_test.astype(int).to_numpy()]
            )

    else:
        MHE_train = np.zeros((df_train.shape[0], 0))
        MHE_test = np.zeros((df_test.shape[0], 0))
        MHE_labels = []

    return MHE_train, MHE_test, MHE_labels


def encode_sentence(model, sentence):
    if sentence:
        word_vectors = [
            model.wv[word] for word in sentence if word in model.wv
        ]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    else:
        word_vectors = np.zeros(model.vector_size)
        return word_vectors


def coding_NLP(df_train_NLP, df_test_NLP, text_enc, model_path):

    # Embeddings with Count Vectorizer
    if text_enc == "CountVectorizer":

        logger.info("   %s : Processing column (Train)" % (text_enc))
        auxList = df_train_NLP.fillna("empty").tolist()

        vectorizer = CountVectorizer(min_df=1)  # , max_features=50)
        vectors_SKL = vectorizer.fit_transform(auxList)  # Sparse matrix
        vectors_SKL_mat = vectors_SKL.todense()

        NLP_train = vectors_SKL_mat

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("empty").tolist()
            vectors_SKL = vectorizer.transform(auxList)  # Sparse matrix
            vectors_SKL_mat = vectors_SKL.todense()

            NLP_test = vectors_SKL_mat
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0], 0))

    # Embeddings with FastText or Word2Vec(Gensim)
    elif text_enc[0:8] in ["FastText", "Word2Vec"]:
        # FastTest works faster if size is multiple of 4
        size_model = int(text_enc[8:])

        tokenizer = ToktokTokenizer()

        logger.info("   {} : Processing (Train)".format(text_enc))

        auxList = df_train_NLP.fillna("").tolist()
        auxToken = [tokenizer.tokenize(doc) for doc in auxList]

        if text_enc[0:8] == "FastText":
            model = gensim.models.FastText(
                min_count=1, vector_size=size_model, workers=1
            )
        else:
            model = gensim.models.Word2Vec(
                min_count=1, vector_size=size_model, workers=1
            )

        model.build_vocab(auxToken, progress_per=100000)

        t = time()
        model.train(
            auxToken,
            total_examples=model.corpus_count,
            epochs=30,
            report_delay=1,
        )
        logger.info(
            "Training time ({} model): {} minutes".format(
                text_enc, round((time() - t) / 60, 2)
            )
        )

        NLP_train = np.array(
            [encode_sentence(model, sentence) for sentence in auxToken]
        )

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("").tolist()
            auxToken = [tokenizer.tokenize(doc) for doc in auxList]
            NLP_test = np.array(
                [encode_sentence(model, sentence) for sentence in auxToken]
            )
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0], 0))

    elif text_enc in [
        "GoogleNews20",
        "GoogleNews128",
        "SWIVEL20",
        "NNLM128",
        "CNNmulti512",
    ]:
        NLP_train = np.array([], dtype=np.float64).reshape(
            df_train_NLP.shape[0], 0
        )
        NLP_test = np.array([], dtype=np.float64).reshape(
            df_test_NLP.shape[0], 0
        )

        batch_size = 4096

        if text_enc == "GoogleNews20":
            embed = hub.load(
                "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
            )
        elif text_enc == "GoogleNews128":
            embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
        elif text_enc == "NNLM128":
            # embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1")
            embed = hub.load(
                "https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2"
            )
        elif text_enc == "CNNmulti512":
            embed = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )
        elif text_enc == "SWIVEL20no":
            # Google Gnews-swivel (Malfunction: different words have the same embeddings)
            embed = hub.load(
                "https://www.kaggle.com/models/google/gnews-swivel/TensorFlow2/tf2-preview-20dim/1"
            )
        elif text_enc == "SWIVEL20":
            # Google Gnews-swivel with pre-built OOV
            embed = hub.load(
                "https://www.kaggle.com/models/google/gnews-swivel/TensorFlow2/tf2-preview-20dim-with-oov/1"
            )

        auxList = df_train_NLP.fillna("").tolist()

        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):
            print("   EMBED {}".format(ii))
            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList) - ii
                break_flag = True

            batch = auxList[ii : ii + batch_size]
            if ii == 0:
                vectors_NLP = embed(batch)
            else:
                vectors_NLP = np.append(vectors_NLP, embed(batch), axis=0)
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train, vectors_NLP])

        logger.info("   %s : Processing column (Test)" % (text_enc))

        if df_test_NLP.shape[0] > 0:

            auxList = df_test_NLP.fillna("empty").tolist()

            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):

                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList) - ii
                    break_flag = True

                batch = auxList[ii : ii + batch_size]
                if ii == 0:
                    vectors_NLP = embed(batch)
                else:
                    vectors_NLP = np.append(vectors_NLP, embed(batch), axis=0)
                if break_flag:
                    break

            NLP_test = np.hstack([NLP_test, vectors_NLP])

    elif text_enc in ["BERT128", "BERT768", "LaBSE"]:
        NLP_train = np.array([], dtype=np.float64).reshape(
            df_train_NLP.shape[0], 0
        )
        NLP_test = np.array([], dtype=np.float64).reshape(
            df_test_NLP.shape[0], 0
        )

        if text_enc == "BERT128":
            batch_size = 128
            preprocessor = hub.KerasLayer(
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
            )
            encoder = hub.KerasLayer(
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2",
                trainable=True,
            )
        elif text_enc == "BERT768":
            batch_size = 128
            preprocessor = hub.KerasLayer(
                "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-cased-preprocess/3"
            )
            encoder = hub.KerasLayer(
                "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-cased-l-12-h-768-a-12/4",
                trainable=True,
            )

        elif text_enc == "LaBSE":
            batch_size = 256
            # preprocessor = hub.KerasLayer(
            #  "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2") # hub.KerasLayer(os.path.join(model_path, "LaBSE_v2_preprocessor"))
            # encoder =  hub.KerasLayer("https://tfhub.dev/google/LaBSE/2") # hub.KerasLayer(os.path.join(model_path, "LaBSE_v2"))
            #
            preprocessor = hub.KerasLayer(
                "https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2"
            )
            encoder = hub.KerasLayer(
                "https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2"
            )
            #
            # preprocessor = hub.KerasLayer(os.path.join(model_path, "LaBSE_v2_preprocessor"))
            # encoder =  hub.KerasLayer(os.path.join(model_path, "LaBSE_v2"))

        logger.info("   %s : Processing column (Train)" % (text_enc))

        auxList = df_train_NLP.fillna("empty").tolist()

        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):

            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList) - ii
                break_flag = True

            batch = auxList[ii : ii + batch_size]
            if ii == 0:
                vectors_NLP = encoder(preprocessor(batch))["default"].numpy()
            else:
                vectors_NLP = np.append(
                    vectors_NLP,
                    encoder(preprocessor(batch))["default"].numpy(),
                    axis=0,
                )
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train, vectors_NLP])

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("empty").tolist()

            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):

                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList) - ii
                    break_flag = True

                batch = auxList[ii : ii + batch_size]
                if ii == 0:
                    vectors_NLP = encoder(preprocessor(batch))[
                        "default"
                    ].numpy()
                else:
                    vectors_NLP = np.append(
                        vectors_NLP,
                        encoder(preprocessor(batch))["default"].numpy(),
                        axis=0,
                    )
                if break_flag:
                    break

            NLP_test = np.hstack([NLP_test, vectors_NLP])

    else:
        raise TypeError("Text encodign not implemented")

    return NLP_train, NLP_test


def coding_NLP2(df_train_NLP, df_test_NLP, text_enc, model_path, subversion):

    # Embeddings with Count Vectorizer
    if text_enc == "CountVectorizer":

        logger.info("   %s : Processing column (Train)" % (text_enc))
        auxList = df_train_NLP.fillna("empty").tolist()

        vectorizer = CountVectorizer(min_df=1)  # , max_features=50)
        vectors_SKL = vectorizer.fit_transform(auxList)  # Sparse matrix
        vectors_SKL_mat = vectors_SKL.todense()

        NLP_train = vectors_SKL_mat

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("empty").tolist()
            vectors_SKL = vectorizer.transform(auxList)  # Sparse matrix
            vectors_SKL_mat = vectors_SKL.todense()

            NLP_test = vectors_SKL_mat
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0], 0))

    # Embeddings with FastText or Word2Vec(Gensim)
    elif text_enc[0:8] in ["FastText", "Word2Vec"]:
        # FastTest works faster if size is multiple of 4
        size_model = int(text_enc[8:])

        tokenizer = ToktokTokenizer()

        logger.info("   {} : Processing (Train)".format(text_enc))

        auxList = df_train_NLP.fillna("").tolist()
        auxToken = [tokenizer.tokenize(doc) for doc in auxList]

        if text_enc[0:8] == "FastText":
            model = gensim.models.FastText(
                min_count=1, vector_size=size_model, workers=1
            )
        else:
            model = gensim.models.Word2Vec(
                min_count=1, vector_size=size_model, workers=1
            )

        model.build_vocab(auxToken, progress_per=100000)

        t = time()
        model.train(
            auxToken,
            total_examples=model.corpus_count,
            epochs=30,
            report_delay=1,
        )
        logger.info(
            "Training time ({} model): {} minutes".format(
                text_enc, round((time() - t) / 60, 2)
            )
        )

        NLP_train = np.array(
            [encode_sentence(model, sentence) for sentence in auxToken]
        )

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("").tolist()
            auxToken = [tokenizer.tokenize(doc) for doc in auxList]
            NLP_test = np.array(
                [encode_sentence(model, sentence) for sentence in auxToken]
            )
        else:
            NLP_test = np.zeros((df_test_NLP.shape[0], 0))

    elif text_enc in [
        "GoogleNews20",
        "GoogleNews128",
        "SWIVEL20",
        "NNLM128",
        "CNNmulti512",
    ]:
        NLP_train = np.array([], dtype=np.float64).reshape(
            df_train_NLP.shape[0], 0
        )
        NLP_test = np.array([], dtype=np.float64).reshape(
            df_test_NLP.shape[0], 0
        )

        batch_size = 4096

        if text_enc == "GoogleNews20":
            embed = hub.load(
                "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
            )
        elif text_enc == "GoogleNews128":
            embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
        elif text_enc == "NNLM128":
            # embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1")
            embed = hub.load(
                "https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2"
            )
        elif text_enc == "CNNmulti512":

            embed = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )
            tf.saved_model.save(
                embed, os.path.join(model_path, "cnn512_" + subversion)
            )

            embed = tf.saved_model.load(
                os.path.join(model_path, "cnn512_" + subversion)
            )
        elif text_enc == "SWIVEL20no":
            # Google Gnews-swivel (Malfunction: different words have the same embeddings)
            embed = hub.load(
                "https://www.kaggle.com/models/google/gnews-swivel/TensorFlow2/tf2-preview-20dim/1"
            )
        elif text_enc == "SWIVEL20":
            # Google Gnews-swivel with pre-built OOV

            embed = hub.load(
                "https://www.kaggle.com/models/google/gnews-swivel/TensorFlow2/tf2-preview-20dim-with-oov/1"
            )
            tf.saved_model.save(
                embed, os.path.join(model_path, "swivel20_" + subversion)
            )

            embed = tf.saved_model.load(
                os.path.join(model_path, "swivel20_" + subversion)
            )

        auxList = df_train_NLP.fillna("").tolist()

        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):
            print("   EMBED {}".format(ii))
            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList) - ii
                break_flag = True

            batch = auxList[ii : ii + batch_size]
            if ii == 0:
                vectors_NLP = embed(batch)
            else:
                vectors_NLP = np.append(vectors_NLP, embed(batch), axis=0)
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train, vectors_NLP])

        logger.info("   %s : Processing column (Test)" % (text_enc))

        if df_test_NLP.shape[0] > 0:

            auxList = df_test_NLP.fillna("empty").tolist()

            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):

                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList) - ii
                    break_flag = True

                batch = auxList[ii : ii + batch_size]
                if ii == 0:
                    vectors_NLP = embed(batch)
                else:
                    vectors_NLP = np.append(vectors_NLP, embed(batch), axis=0)
                if break_flag:
                    break

            NLP_test = np.hstack([NLP_test, vectors_NLP])

    elif text_enc in ["BERT128", "BERT768", "LaBSE"]:
        NLP_train = np.array([], dtype=np.float64).reshape(
            df_train_NLP.shape[0], 0
        )
        NLP_test = np.array([], dtype=np.float64).reshape(
            df_test_NLP.shape[0], 0
        )

        if text_enc == "BERT128":
            batch_size = 128
            preprocessor = hub.KerasLayer(
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
            )
            encoder = hub.KerasLayer(
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2",
                trainable=True,
            )
        elif text_enc == "BERT768":
            batch_size = 128
            preprocessor = hub.KerasLayer(
                "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-cased-preprocess/3"
            )
            encoder = hub.KerasLayer(
                "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-cased-l-12-h-768-a-12/4",
                trainable=True,
            )

        elif text_enc == "LaBSE":
            batch_size = 256
            # preprocessor = hub.KerasLayer(
            #  "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2") # hub.KerasLayer(os.path.join(model_path, "LaBSE_v2_preprocessor"))
            # encoder =  hub.KerasLayer("https://tfhub.dev/google/LaBSE/2") # hub.KerasLayer(os.path.join(model_path, "LaBSE_v2"))
            #
            preprocessor = hub.KerasLayer(
                "https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-multilingual-preprocess/2"
            )
            encoder = hub.KerasLayer(
                "https://www.kaggle.com/models/google/labse/TensorFlow2/labse/2"
            )
            #
            # preprocessor = hub.KerasLayer(os.path.join(model_path, "LaBSE_v2_preprocessor"))
            # encoder =  hub.KerasLayer(os.path.join(model_path, "LaBSE_v2"))

        logger.info("   %s : Processing column (Train)" % (text_enc))

        auxList = df_train_NLP.fillna("empty").tolist()

        break_flag = False
        for ii in tqdm(range(0, len(auxList), batch_size)):

            if (ii + batch_size) > len(auxList):
                batch_size = len(auxList) - ii
                break_flag = True

            batch = auxList[ii : ii + batch_size]
            if ii == 0:
                vectors_NLP = encoder(preprocessor(batch))["default"].numpy()
            else:
                vectors_NLP = np.append(
                    vectors_NLP,
                    encoder(preprocessor(batch))["default"].numpy(),
                    axis=0,
                )
            if break_flag:
                break

        NLP_train = np.hstack([NLP_train, vectors_NLP])

        logger.info("   %s : Processing column (Test)" % (text_enc))
        if df_test_NLP.shape[0] > 0:
            auxList = df_test_NLP.fillna("empty").tolist()

            break_flag = False
            for ii in tqdm(range(0, len(auxList), batch_size)):

                if (ii + batch_size) > len(auxList):
                    batch_size = len(auxList) - ii
                    break_flag = True

                batch = auxList[ii : ii + batch_size]
                if ii == 0:
                    vectors_NLP = encoder(preprocessor(batch))[
                        "default"
                    ].numpy()
                else:
                    vectors_NLP = np.append(
                        vectors_NLP,
                        encoder(preprocessor(batch))["default"].numpy(),
                        axis=0,
                    )
                if break_flag:
                    break

            NLP_test = np.hstack([NLP_test, vectors_NLP])

    else:
        raise TypeError("Text encodign not implemented")

    return NLP_train, NLP_test


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def coding_columns(
    df_train,
    df_test,
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
        Type of binary encoding ("OneZero", "PlusMinusOne")
    numeric_feats : list of str
        List with numeric features in df_train and df_test
    normalization : str
        Type of normalization used for numeric features ("Standard", "MinMax", "MinMaxOne", "MaxAbs")
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
    NUM_train, NUM_test, NUM_labels = coding_NUM(
        df_train, df_test, numeric_feats, normalization
    )

    # Binary features
    logger.info("  *  Encoding binary features: {}".format(binary_feats))
    BIN_train, BIN_test, BIN_labels = coding_BIN(
        df_train, df_test, binary_feats, binary_type
    )

    # One-Hot-Encoding (OHE) of all the data sets
    logger.info(
        "  *  Encoding categorical features: {}".format(
            purely_categorical_feats
        )
    )
    OHE_train, OHE_test, OHE_labels = coding_OHE(
        df_train, df_test, purely_categorical_feats
    )

    # Multiple-Hot-Encoding (MHE)
    logger.info(
        "  *  Encoding multi-categorical features: {}".format(
            multiple_categorical_feats
        )
    )
    MHE_train, MHE_test, MHE_labels = coding_MHE(
        df_train, df_test, multiple_categorical_feats
    )

    # Encode NLP features
    logger.info("  *  Encoding text features (NLP): {}".format(nlp_feats))
    # NLP_train_list = []
    # NLP_test_list = []
    NLP_train = np.zeros((df_train.shape[0], 0))
    NLP_test = np.zeros((df_test.shape[0], 0))
    for idx, feat_list in enumerate(nlp_feats):
        logger.info(
            "     + Encoding jointly {} columns: {}".format(
                len(feat_list), feat_list
            )
        )
        # Use the last specified encoding for additional features
        text_enc_type = text_enc[idx] if idx < len(text_enc) else text_enc[-1]

        # joint_text_train = df_train[feat_list].astype(str).apply(' '.join)
        joint_text_train = (
            df_train[feat_list].fillna("").astype(str).agg(" ".join, axis=1)
        )
        joint_text_test = (
            df_test[feat_list].fillna("").astype(str).agg(" ".join, axis=1)
        )

        NLP_train_feat_cur, NLP_test_feat_cur = coding_NLP(
            joint_text_train, joint_text_test, text_enc_type, model_path
        )
        NLP_train = np.hstack([NLP_train, NLP_train_feat_cur])
        NLP_test = np.hstack([NLP_test, NLP_test_feat_cur])

    # Labels of encoded characteristics
    coding_labels = NUM_labels + BIN_labels + OHE_labels + MHE_labels

    # Train and test patterns and labels
    x_train = np.hstack(
        [NUM_train, BIN_train, OHE_train, MHE_train, NLP_train]
    )
    if "Label" in df_train.columns:
        y_train = df_train["Label"].replace({True: 1, False: 0}).to_numpy()
    else:
        y_train = 0

    if df_test.shape[0] > 0:
        x_test = np.hstack([NUM_test, BIN_test, OHE_test, MHE_test, NLP_test])
        if "Label" in df_test.columns:
            y_test = df_test["Label"].replace({True: 1, False: 0}).to_numpy()
        else:
            y_test = 0

    else:
        x_test = 0
        y_test = 0

    return x_train, y_train, x_test, y_test, coding_labels


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def coding_columns2(
    df_train,
    df_test,
    binary_feats,
    bincat_dict,
    binary_type,
    numeric_feats,
    normalization,
    purely_categorical_feats,
    multiple_categorical_feats,
    nlp_feats,
    text_enc,
    model_path,
    subversion,
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
        Type of binary encoding ("OneZero", "PlusMinusOne")
    numeric_feats : list of str
        List with numeric features in df_train and df_test
    normalization : str
        Type of normalization used for numeric features ("Standard", "MinMax", "MinMaxOne", "MaxAbs")
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
    NUM_train, NUM_test, NUM_labels = coding_NUM2(
        df_train, df_test, numeric_feats, normalization, subversion
    )

    with open("config/inference_config_" + subversion + ".yaml", "r") as file:
        yaml_dict = yaml.safe_load(file)

    yaml_dict["params"]["nlp_feats"] = nlp_feats
    yaml_dict["params"]["text_enc"] = text_enc

    with open("config/inference_config_" + subversion + ".yaml", "w") as file:
        yaml.dump(yaml_dict, file, default_flow_style=False)

    # Binary features
    logger.info("  *  Encoding binary features: {}".format(binary_feats))
    BIN_train, BIN_test, BIN_labels = coding_BIN2(
        df_train, df_test, binary_feats, bincat_dict, binary_type, subversion
    )

    # One-Hot-Encoding (OHE) of all the data sets
    logger.info(
        "  *  Encoding categorical features: {}".format(
            purely_categorical_feats
        )
    )
    OHE_train, OHE_test, OHE_labels = coding_OHE2(
        df_train, df_test, purely_categorical_feats, subversion
    )

    # Multiple-Hot-Encoding (MHE)
    logger.info(
        "  *  Encoding multi-categorical features: {}".format(
            multiple_categorical_feats
        )
    )
    MHE_train, MHE_test, MHE_labels = coding_MHE(
        df_train, df_test, multiple_categorical_feats
    )

    # Encode NLP features
    logger.info("  *  Encoding text features (NLP): {}".format(nlp_feats))
    # NLP_train_list = []
    # NLP_test_list = []
    NLP_train = np.zeros((df_train.shape[0], 0))
    NLP_test = np.zeros((df_test.shape[0], 0))
    for idx, feat_list in enumerate(nlp_feats):
        logger.info(
            "     + Encoding jointly {} columns: {}".format(
                len(feat_list), feat_list
            )
        )
        # Use the last specified encoding for additional features
        text_enc_type = text_enc[idx] if idx < len(text_enc) else text_enc[-1]

        # joint_text_train = df_train[feat_list].astype(str).apply(' '.join)
        joint_text_train = (
            df_train[feat_list].fillna("").astype(str).agg(" ".join, axis=1)
        )
        joint_text_test = (
            df_test[feat_list].fillna("").astype(str).agg(" ".join, axis=1)
        )

        NLP_train_feat_cur, NLP_test_feat_cur = coding_NLP2(
            joint_text_train,
            joint_text_test,
            text_enc_type,
            model_path,
            subversion,
        )
        NLP_train = np.hstack([NLP_train, NLP_train_feat_cur])
        NLP_test = np.hstack([NLP_test, NLP_test_feat_cur])

    # Labels of encoded characteristics
    coding_labels = NUM_labels + BIN_labels + OHE_labels + MHE_labels

    # Train and test patterns and labels
    x_train = np.hstack(
        [NUM_train, BIN_train, OHE_train, MHE_train, NLP_train]
    )
    if "Label" in df_train.columns:
        # y_train = df_train["Label"].replace({True: 1, False: 0}).to_numpy()
        # Explicitly convert boolean to integers after replacement
        y_train = df_train["Label"].replace({True: 1, False: 0}).astype(int).to_numpy()
    else:
        y_train = np.array([0]) # 0

    if df_test.shape[0] > 0:
        x_test = np.hstack([NUM_test, BIN_test, OHE_test, MHE_test, NLP_test])
        if "Label" in df_test.columns:
            # y_test = df_test["Label"].replace({True: 1, False: 0}).to_numpy()
            # Explicitly convert boolean to integers after replacement
            y_test = df_test["Label"].replace({True: 1, False: 0}).astype(int).to_numpy()

        else:
            y_test = np.array([0]) # 0

    else:
        x_test = np.array([0]) # 0
        y_test = np.array([0]) # 0

    return x_train, y_train, x_test, y_test, coding_labels


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_encoding(
    df_alarms,
    features,
    binary_type="PlusMinusOne",
    normalization="Standard",
    split_idx="random",
    num_folds=5,
    text_enc="CNNmulti512",
    feats2keep=[],
    data_out_path=None,
    model_path=None,
):
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    logger.info("Starting encoding")
    (
        binary_feats,
        numeric_feats,
        purely_categorical_feats,
        multiple_categorical_feats,
        nlp_feats,
    ) = features

    if type(split_idx) == str:
        split_idx_dict = func_idx_train_test(
            df_alarms, num_folds=num_folds, split_idx_type=split_idx
        )
    else:
        split_idx_dict = split_idx
        num_folds = int(len(split_idx_dict) / 2)

    if df_alarms.index.max() > (df_alarms.shape[0] - 1):
        df_alarms.reset_index(inplace=True, drop=True)

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    idx_test_list = []
    df_train_list = []
    df_test_list = []
    coding_labels_C = []

    for kFold in range(num_folds):
        logger.info("  * Encoding Fold {}".format(kFold))

        df_train = df_alarms.loc[split_idx_dict["train{}".format(kFold)]]
        df_test = df_alarms.loc[split_idx_dict["test{}".format(kFold)]]

        x_train, y_train, x_test, y_test, coding_labels = coding_columns(
            df_train,
            df_test,
            binary_feats,
            binary_type,
            numeric_feats,
            normalization,
            purely_categorical_feats,
            multiple_categorical_feats,
            nlp_feats,
            text_enc,
            model_path,
        )

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        idx_test_list.append(split_idx_dict["test{}".format(kFold)])
        df_train_list.append(df_train[feats2keep])
        df_test_list.append(df_test[feats2keep])
        coding_labels_C.append(coding_labels)

        logger.info(
            "   + Shape of training set (fold {}) is {}".format(
                kFold, x_train.shape
            )
        )
        logger.info(
            "   + Shape of test set (fold {}) is {}".format(
                kFold, x_test.shape
            )
        )

    coded_data = [
        x_train_list,
        x_test_list,
        y_train_list,
        y_test_list,
        idx_test_list,
        coding_labels_C,
        nlp_feats,
        df_train_list,
        df_test_list,
        binary_type,
        normalization,
        text_enc,
    ]

    if data_out_path != None:
        with open(data_out_path, "wb") as f:  # Python 3: open(..., "wb")
            pickle.dump(coded_data, f)

        logger.info("Coded dataframe saved to: {}".format(data_out_path))

    return coded_data


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_encoding2(
    df_alarms,
    features,
    bincat_dict,
    binary_type="PlusMinusOne",
    normalization="Standard",
    split_idx="random",
    num_folds=5,
    text_enc="CNNmulti512",
    feats2keep=[],
    data_out_path=None,
    model_path=None,
    subversion="prueba",
):
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    logger.info("Starting encoding")
    (
        binary_feats,
        numeric_feats,
        purely_categorical_feats,
        multiple_categorical_feats,
        nlp_feats,
    ) = features

    if type(split_idx) == str:
        split_idx_dict = func_idx_train_test(
            df_alarms, num_folds=num_folds, split_idx_type=split_idx
        )
    else:
        split_idx_dict = split_idx
        num_folds = int(len(split_idx_dict) / 2)

    if df_alarms.index.max() > (df_alarms.shape[0] - 1):
        df_alarms.reset_index(inplace=True, drop=True)

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    idx_test_list = []
    df_train_list = []
    df_test_list = []
    coding_labels_C = []

    for kFold in range(num_folds):
        logger.info("  * Encoding Fold {}".format(kFold))

        df_train = df_alarms.loc[split_idx_dict["train{}".format(kFold)]]
        df_test = df_alarms.loc[split_idx_dict["test{}".format(kFold)]]

        x_train, y_train, x_test, y_test, coding_labels = coding_columns2(
            df_train,
            df_test,
            binary_feats,
            bincat_dict,
            binary_type,
            numeric_feats,
            normalization,
            purely_categorical_feats,
            multiple_categorical_feats,
            nlp_feats,
            text_enc,
            model_path,
            subversion,
        )

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        idx_test_list.append(split_idx_dict["test{}".format(kFold)])
        df_train_list.append(df_train[feats2keep])
        df_test_list.append(df_test[feats2keep])
        coding_labels_C.append(coding_labels)

        logger.info(
            "   + Shape of training set (fold {}) is {}".format(
                kFold, x_train.shape
            )
        )
        logger.info(
            "   + Shape of test set (fold {}) is {}".format(
                kFold, x_test.shape
            )
        )

    coded_data = [
        x_train_list,
        x_test_list,
        y_train_list,
        y_test_list,
        idx_test_list,
        coding_labels_C,
        nlp_feats,
        df_train_list,
        df_test_list,
        binary_type,
        normalization,
        text_enc,
    ]

    if data_out_path != None:
        with open(data_out_path, "wb") as f:  # Python 3: open(..., "wb")
            pickle.dump(coded_data, f)

        logger.info("Coded dataframe saved to: {}".format(data_out_path))

    return coded_data


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def func_idx_train_test(
    labels,
    split_idx_type: str = "stratified",
    num_folds: int = 5,
    split_idx_path: str = None,
    test_size: float = 0.2,
) -> Dict:
    """Create a train/test split based on parameters.

    Args:
        labels (_type_): Labels for the split. "stratified" accepts pd.Series, list and np.ndarray, while the other options also accept the number of labels (int).
        split_idx_type (str, optional): "stratified": Standard CV, "random": Random CV, and "sequential": Sequential split. Defaults to "stratified".
        num_folds (int, optional): Number of folds for CV. Only used in "stratified" and "random". Defaults to 5.
        split_idx_path (str, optional): Path to save split indexes. Defaults to None.
        test_size (float, optional): Test set size. Defaults to 0.2.

    Returns:
        Dict: Dictionary with Train and test indexes.
    """
    logger.info("Train / Test split")
    # import the dictionaries with train and text indices
    if split_idx_type == "stratified":

        if type(labels) in [list, np.ndarray]:
            labels = pd.Series(labels)

        logger.info("  * Stratified train / test split")
        list_index_True = list(labels[labels == True].index)
        list_index_False = list(labels[labels == False].index)

        random.shuffle(list_index_True)
        random.shuffle(list_index_False)

        size_fold_True = len(list_index_True) // num_folds
        size_fold_False = len(list_index_False) // num_folds

        size_folds_True = [
            (
                size_fold_True + 1
                if x < (len(list_index_True) % size_fold_True)
                else size_fold_True
            )
            for x in range(num_folds)
        ]
        size_folds_False = [
            (
                size_fold_False + 1
                if x < (len(list_index_False) % size_fold_False)
                else size_fold_False
            )
            for x in range(num_folds)
        ]

        index_folds_True = []
        index_folds_False = []

        split_idx_dict = {}

        index_total_set = set(range(labels.shape[0]))

        for kFold in range(num_folds):
            index_folds_True.append(
                list_index_True[
                    sum(size_folds_True[0:kFold]) : sum(
                        size_folds_True[0 : kFold + 1]
                    )
                ]
            )
            index_folds_False.append(
                list_index_False[
                    sum(size_folds_False[0:kFold]) : sum(
                        size_folds_False[0 : kFold + 1]
                    )
                ]
            )

        for kFold in range(num_folds):

            index_test = index_folds_True[kFold] + index_folds_False[kFold]
            # index_train = [x for x in range(df_alarms.shape[0]) if x not in  index_test]
            index_train = list(index_total_set - set(index_test))

            split_idx_dict["train{}".format(kFold)] = index_train
            split_idx_dict["test{}".format(kFold)] = index_test

    elif split_idx_type == "random":

        if type(labels) == int:
            num_patterns = labels
        else:
            num_patterns = len(labels)

        logger.info("  * Random train / test split")
        list_index = list(range(num_patterns))

        random.shuffle(list_index)

        size_fold = len(list_index) // num_folds

        size_folds = [
            size_fold + 1 if x < (len(list_index) % size_fold) else size_fold
            for x in range(num_folds)
        ]

        # index_folds = []
        print(size_folds)

        split_idx_dict = {}

        index_total_set = set(range(num_patterns))

        for kFold in range(num_folds):
            # index_folds.append(list_index[sum(size_folds[0:kFold]):sum(size_folds[0:kFold+1])])
            logger.info("   FOLD----")
            index_test = set(
                list_index[
                    sum(size_folds[0:kFold]) : sum(size_folds[0 : kFold + 1])
                ]
            )
            index_train = index_total_set - index_test

            split_idx_dict["train{}".format(kFold)] = list(index_train)
            split_idx_dict["test{}".format(kFold)] = list(index_test)

    elif split_idx_type == "sequential":
        if type(labels) == int:
            num_patterns = labels
        else:
            num_patterns = len(labels)

        list_index = list(range(num_patterns))

        split_idx_dict = {
            "train0": list_index[0 : int(num_patterns * (1 - test_size))],
            "test0": list_index[int(num_patterns * (1 - test_size)) :],
        }

    if split_idx_path != None:
        logger.info(
            "  * Saving Train / Test split to {}".format(split_idx_path)
        )
        with open(split_idx_path, "w") as fp:
            json.dump(split_idx_dict, fp, indent=2)

    return split_idx_dict
