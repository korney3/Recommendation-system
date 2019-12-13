import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from lightfm.data import Dataset
from lightfm import LightFM


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import os

from scipy.sparse import coo_matrix

import time
import requests

import urllib
from urllib.request import Request
import json

import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def Hybrid_Rec_System_Prediction(cust_ids=None,num_support_topics=10, num_predict_topics=10,print_flag=True, n_components=100, n_epochs=30):
    """ Predict list of topic relevant to customer id

    Parameters
    ----------
    cust_ids : int, list of ints, default None
        Id of interaction between customer and topic from 'customer_id_range' column

    num_support_topics : int
        Number of topics already approved by customer used for prediction of new ones
    num_predict_topics : int
        Max number of topics to recommend

    print_flag : bool
        Print info about customer and categories in a human-readible form

    n_components : int
        Size of vector embedding in LightFM model
    n_epochs : int
        Number of epochs to train model
    """
    def load_data():
        """ Load cutomers and transactions dataset

        Returns
        -------
        new_cust : pandas.DataFrame
            DataFrame with info from customer (id, gender, marital_status, etc..) in preprocessed numerical form
        trans_items : pandas.DataFrame
            DataFrame with synchronized info from interaction between customer and particular topic (customer_id, mcc_id,weight)
        mcc_code : pandas.DataFrame
            Connection between topics and their mcc code    
        """
        request = Request('https://raw.githubusercontent.com/Kichkun/BugMakers/master/categories_map.csv?token=AK525P2DNTT7E7DQIQRLIWS56UXUS')
        with urllib.request.urlopen(request) as f:
            mcc_codes = pd.read_csv(f)

        request = Request('http://hackathon.oooo.su/api/service/alisa')
        with urllib.request.urlopen(request) as f:
            new_cust = pd.read_csv(f)
        request = Request('http://hackathon.oooo.su/api/data/weights')
        with urllib.request.urlopen(request) as f:
            trans_items = pd.read_csv(f)

        new_cust = new_cust[new_cust.customer_id.isin(trans_items.customer_id)]
        print('Number of users in system:')
        print(new_cust.customer_id.nunique())
        return new_cust, trans_items, mcc_codes

    def data_preprocessing(new_cust, trans_items):
        """ Load cutomers and transactions dataset
        Parameters
        ----------
        new_cust : pandas.DataFrame
            DataFrame with info from customer (id, gender, marital_status, etc..) in preprocessed numerical form
        trans_items : pandas.DataFrame
            DataFrame with synchronized info from interaction between customer and particular topic (customer_id, mcc_id,weight)
        Returns
        -------
        df_customers : pandas.DataFrame
            DataFrame with info from customer (id, gender, marital_status, etc..) in preprocessed numerical form suitable for LightFM
        df_mcc : pandas.DataFrame
            DataFrame with synchronized info from interaction between customer and particular topic (customer_id, mcc_id,weight) suitable for LightFM
        df_merge : pandas.DataFrame
            DataFrame with synchronized info from customer and transaction DataFrames suitable for LightFM
        customer_feature_list : LightFM module
            User features in  a LightFM format   
        """
        def generate_int_id(dataframe, id_col_name):
            new_dataframe=dataframe.assign(
                int_id_col_name=np.arange(len(dataframe))
                ).reset_index(drop=True)
            return new_dataframe.rename(columns={'int_id_col_name': id_col_name})

        def create_features(dataframe, features_name, id_col_name):
            features = dataframe[features_name].apply(
                lambda x: ','.join(x.map(str)), axis=1)
            features = features.str.split(',')
            features = list(zip(dataframe[id_col_name], features))
            return features

        def generate_feature_list(dataframe, features_name):
            features = dataframe[features_name].apply(
                lambda x: ','.join(x.map(str)), axis=1)
            features = features.str.split(',')
            features = features.apply(pd.Series).stack().reset_index(drop=True)
            return features

        df_customers = generate_int_id(new_cust, 'customer_id_range')
        df_mcc = generate_int_id(trans_items, 'merchant_mcc_range')
        df_merge=df_mcc.merge(df_customers, how='inner',left_on='customer_id',right_on='customer_id')
        feats = [feat for feat in list(df_customers) if ('product_' not in feat and 'customer_id' not in feat)]

        customer_feature_list = generate_feature_list(
            df_customers,
            feats)
        df_customers['cust_features'] = create_features(
            df_customers,
            feats,
            'customer_id_range')
        return df_customers,df_mcc, df_merge, customer_feature_list

    def LightFM_dataset_creation(df_customers,df_mcc, df_merge, customer_feature_list):
        dataset = Dataset()
        dataset.fit(
            set(df_customers['customer_id_range']), 
            set(df_mcc['merchant_mcc_range']),
            user_features=customer_feature_list)

        df_merge['cust_merch_tuple'] = list(zip(
            df_merge.customer_id_range, df_merge.merchant_mcc_range, df_merge.weight))

        interactions, weights = dataset.build_interactions(
            df_merge['cust_merch_tuple'])

        customer_features = dataset.build_user_features(
            df_customers['cust_features'])

        return df_merge,interactions, weights,customer_features

    def LightFM_model_train(df_merge,interactions, weights,customer_features, n_components=100, n_epochs=30):
        model = LightFM( 
        no_components=n_components,
        learning_rate=0.05,
        loss='warp',
        random_state=2019)

        model.fit(
            interactions,
            user_features=customer_features, sample_weight=weights,
            epochs=n_epochs, num_threads=4, verbose=True)
        return model

    def recommend_topics(customer_ids,num_of_prev_questions=3, num_of_predicts=8,print_flag=True):
        """ Predict list of topic relevant to customer id

        Parameters
        ----------
        Customer : int
            Id of interaction between customer and topic from 'customer_id_range' column

        df_customers : pandas.DataFrame
            DataFrame with info from customer (id, gender, marital_status, etc..) in preprocessed numerical form
        df_mcc : pandas.DataFrame
            DataFrame with synchronized info from interaction between customer and particular topic (customer_id, mcc_id,weight)
        df_merge : pandas.DataFrame
            DataFrame with synchronized info from customer and transaction DataFrames
        mcc_code : pandas.DataFrame
            Connection between topics and their mcc code

        model : LightFM model
            Recommendation model from LightFm library, trained with whole database
        customer_features : LightFM module
            User features in  a LightFM format

        num_of_prev_questions : int
            Number of topics already approved by customer used for prediction of new ones
        num_of_predicts : int
            Max number of topics to recommend

        print_flag : bool
            Print info about customer and categories in a human-readible form


        Returns
        -------
        res : numpy.array
            Array of topic's idea relevant to the customer
        """
        res=[]
        f=open('detailed_recommendation','w')
        f.close()
        for i,customer in tqdm(enumerate(customer_ids)):
            previous_q_id_num = df_merge.loc[df_merge['customer_id_range'] == customer][:num_of_prev_questions+1]['merchant_mcc_range']
            df_previous_questions = df_mcc.loc[df_mcc['merchant_mcc_range'].isin(previous_q_id_num)]
            if print_flag:
                print('Customer Id (' + str(customer) + ")")
            with open('detailed_recommendation','a') as f:
                f.write('Customer Id (' + str(customer) + ")"+'\n')
            info_cust=df_customers.loc[df_customers['customer_id_range']==customer]
            start=df_previous_questions[['merchant_mcc']].merge(mcc_codes[['merchant_mcc','edited_description']], on='merchant_mcc')
            if int(info_cust['gender_cd'])==1:
                if print_flag:
                    print('Gender: M')
                with open('detailed_recommendation','a') as f:
                    f.write('Пол: M'+'\n')
            else:
                if print_flag:
                    print('Gender: F')
                with open('detailed_recommendation','a') as f:
                    f.write('Пол: Ж'+'\n')    
            if print_flag:
                print('Age: '+str(info_cust['age'].values[0]))
                print('Number of children: '+str(info_cust['children_cnt'].values[0]))
                print(
                start.drop_duplicates())
                
            with open('detailed_recommendation','a') as f:
                    f.write('Возраст: '+str(info_cust['age'].values[0])+'\n')
            with open('detailed_recommendation','a') as f:
                    start.drop_duplicates().to_string(f)
                    f.write('\n')


            discard_qu_id = df_previous_questions['merchant_mcc_range'].values.tolist()
            df_use_for_prediction = df_mcc.loc[~df_mcc['merchant_mcc_range'].isin(discard_qu_id)]
            questions_id_for_predict = df_use_for_prediction['merchant_mcc_range'].values.tolist()#[:200]


            scores = model.predict(
                customer,
                questions_id_for_predict,
                user_features=customer_features)

            df_use_for_prediction['scores'] = scores
            df_use_for_prediction = df_use_for_prediction.sort_values(by='scores', ascending=False)[:num_of_predicts+1]
            if print_flag:
                print('Customer Id (' + str(customer) + "): Recommended Topics: ")
            with open('detailed_recommendation','a') as f:
                    f.write('Customer Id (' + str(customer) + "): Recommended Topics: "+'\n')
            fin=df_use_for_prediction[['merchant_mcc']].merge(mcc_codes[['merchant_mcc','edited_description']], on='merchant_mcc')
            fin=fin.loc[~fin['merchant_mcc'].isin(start['merchant_mcc'])]
            if print_flag:
                print(fin.drop_duplicates())
                
            with open('detailed_recommendation','a') as f:
                fin.drop_duplicates().to_string(f)
                f.write('\n')
            res.append(fin['merchant_mcc'].values)

        return res 

    new_cust, trans_items, mcc_codes = load_data()

    df_customers,df_mcc, df_merge, customer_feature_list = data_preprocessing(new_cust, trans_items)
    if cust_ids is None:
        cust_ids=df_customers['customer_id_range']
    df_merge,interactions, weights,customer_features = LightFM_dataset_creation(df_customers,df_mcc, df_merge, customer_feature_list)

    model = LightFM_model_train(df_merge,interactions, weights,customer_features, n_components=n_components, n_epochs=n_epochs)
    print('done train')

    res=recommend_topics(customer_ids=cust_ids,num_of_prev_questions=num_support_topics, num_of_predicts=num_predict_topics,print_flag=print_flag)
    
    output_dict={}
    for i in range(len(cust_ids)):
        cust_real_id = df_customers[df_customers.customer_id_range==cust_ids[i]].customer_id.values[0]
        output_dict[str(cust_real_id)]=[str(x) for x in res[i]]
        if i%1000==0:
            print(i)
    with open('customer_topic.json','w') as f:
        json.dump(output_dict,f)


    return 

if __name__ == "__main__":
    res=Hybrid_Rec_System_Prediction(cust_ids=[1,1023,41345], num_support_topics=10, num_predict_topics=10,print_flag=True, n_components=100, n_epochs=10)
    print(res)
