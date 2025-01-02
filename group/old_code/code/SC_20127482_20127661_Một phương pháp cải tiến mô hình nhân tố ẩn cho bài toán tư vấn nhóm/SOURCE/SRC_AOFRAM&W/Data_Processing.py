import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import statistics
from fractions import Fraction as F
from decimal import Decimal as D
import warnings
warnings.filterwarnings('ignore')
import csv
import time
import os
from numpy.random import choice
from sklearn.model_selection import train_test_split
import torch.cuda

class Data_Processing:
    def __init__ (self, file_rating,fname_train,fname_test,test_ratio):
        
        # df_ratings = pd.read_csv(
        #     os.path.join('./data/Digital_Music.csv'),
        #     usecols=['user_id', 'item_id','avg_rating'],
        #     dtype={'user_id':'string', 'item_id':'string','avg_rating':'float32'})
        # df_copy = df_ratings.copy()

        # user_id_mapping = {user_id: i+1 for i, user_id in enumerate(df_copy['user_id'].unique())}
        # item_id_mapping = {item_id: i+1 for i, item_id in enumerate(df_copy['item_id'].unique())}

        # df_copy['user_id'] = df_copy['user_id'].map(user_id_mapping)
        # df_copy['item_id'] = df_copy['item_id'].map(item_id_mapping)

        # shape = df_copy.shape
        # num_rows = shape[0]
        # num_columns = shape[1]
        # print(f'Số hàng: {num_rows}')
        # print(f'Số cột: {num_columns}')

        # dffile = pd.DataFrame(df_copy)
        # fname = './Data/Digital_Music_pro.csv'
        # dffile.to_csv(fname, index = None)

        self.data = pd.read_csv(file_rating)
        train,test = train_test_split(self.data, test_size= test_ratio, random_state=42)
        # #save the data
        train.to_csv(fname_train, index=False)
        test.to_csv(fname_test, index=False)

        df = pd.read_csv('train_data.csv')
        #df = df.drop(columns=['textRating'])
        #df = df.drop(columns=['rating'])
        df.to_csv('train_data.csv', index=False)

        df = pd.read_csv('test_data.csv')
        #df = df.drop(columns=['textRating'])
        #df = df.drop(columns=['avg_rating'])
        df.to_csv('test_data.csv', index=False)

        self.df_ratings = pd.read_csv(
            os.path.join(file_rating),
            usecols=['user_id', 'item_id','avg_rating'],
            dtype={'user_id':'int32', 'item_id':'int32','avg_rating':'float32'})

        # ,'rating','textRating'
        # ,'rating':'float32','textRating':'float32'
        # self.df_copy = df_ratings.copy()

        # user_id_mapping = {user_id: i+1 for i, user_id in enumerate(self.df_copy['user_id'].unique())}
        # item_id_mapping = {item_id: i+1 for i, item_id in enumerate(self.df_copy['item_id'].unique())}

        # self.df_copy['user_id'] = self.df_copy['user_id'].map(user_id_mapping)
        # self.df_copy['item_id'] = self.df_copy['item_id'].map(item_id_mapping)

        # #print("Duplicated:",self.df_copy.index.duplicated().any())
        # print(self.df_copy)
        # shape = self.df_copy.shape
        # num_rows = shape[0]
        # num_columns = shape[1]

        # print(f'Số hàng: {num_rows}')
        # print(f'Số cột: {num_columns}')
        df_features = self.df_ratings.pivot_table(
            index='user_id',
            columns='item_id',
            values='avg_rating'
        ).fillna(0)
        self.R_matrix = df_features.to_numpy()

        self.n_users = int(np.size(self.R_matrix, 0)) 
        self.n_items = int(np.size(self.R_matrix, 1))
        print(self.n_users,"-",self.n_items)
        # matrix_dict = {}
        # for row in self.df_ratings.itertuples(index=False):
        #     item_id, user_id, rating = row
        #     if user_id not in matrix_dict:
        #         matrix_dict[user_id] = {}
        #     matrix_dict[user_id][item_id] = rating

        # matrix = pd.DataFrame(matrix_dict).fillna(0)
        # print(matrix)

    def Find_Influ(self, fnameInflu):
        self.Influ = np.zeros(self.n_users)
        for i in range(self.n_users):
            temp1 = list(self.R_matrix[i,:])
            self.Influ[i] = self.n_items - temp1.count(0)
        Inf_pra = pd.DataFrame(self.Influ, columns =['Influ'])
        Inf_pra.to_csv(fnameInflu, index = None)

    def Sharing_Group(self, n_group, n_userInGroup, fnameGroup):
        # Tạo nhóm group
        group = choice(self.df_ratings.user_id, size=(n_group, n_userInGroup)) 
        col_sl = np.zeros((n_group,1))
        n= n_group
        i=0
        while i<n:
            list_a = list(group[i,:])
            # hàm set đếm số user phân biệt, nếu trong list_a không có user trùng nhau thì len(Set) = len(list)
            if (len(set(list_a)) == len(list_a)):
                temp1 = np.ones(self.n_items) #tạo vector 1 có n items
                # xét mỗi member trong group
                for a in range (n_userInGroup):
                    user = group[i, a]
                    temp1 = temp1 * self.R_matrix[int(user)-1]
                col_sl[i] = int (self.n_items - list(temp1).count(0))
                i = i+1
            else:
                group = np.delete(group, (i), axis=0)
                col_sl = np.delete(col_sl, (i), axis=0 )
                if (i !=0):
                    i=i-1
                n=n-1
        if (int(np.size(group, 0)) != 0):
            data_temp = np.hstack((group, col_sl))
            Groupn_pra = pd.DataFrame(data_temp, columns = [x + str(y) for x in ['User'] for y in  range(n_userInGroup)] + ['SL item đánh giá chung'])
            Groupn_pra.to_csv(fnameGroup, index = None)



n_userGroup = 4
number_group = 1000

print("Starting...")
fname_data = "./data/Musical_Instrument_rating.csv"
fname_group = 'Group/Group' + str(n_userGroup) +'.csv'
fname_Influ = "FactorMatrix/Influ.csv"

step1 = Data_Processing(fname_data,"train_data.csv","test_data.csv",0.5)
print("Find Influence...")
step1.Find_Influ(fname_Influ)
step1.Sharing_Group(number_group, n_userGroup, fname_group)
print("End")