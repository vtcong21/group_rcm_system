import numpy as np
import pandas as pd
import os
import statistics
from fractions import Fraction as F
from decimal import Decimal as D
import warnings
import csv
import time
import os
from numpy.random import choice
from sklearn.model_selection import train_test_split

class R:
  def __init__ (self, u, i, r ):
    self.user = u
    self.item = i
    self.rating = r

# 32489 item - 319336 user
class Offline:
    def __init__ (self, fname_R , nfactor, anpha, lam, max_iter):
        self.fname_R = fname_R

        df_ratings = pd.read_csv(
            os.path.join(self.fname_R),
            usecols=['item_id', 'user_id', 'avg_rating'],
            dtype={'item_id': 'string', 'user_id': 'string', 'avg_rating': 'float32'})
        
        # df_copy = df_ratings.copy()

        # user_id_mapping = {user_id: i+1 for i, user_id in enumerate(df_copy['user_id'].unique())}
        # item_id_mapping = {item_id: i+1 for i, item_id in enumerate(df_copy['item_id'].unique())}

        df_features = df_ratings.pivot_table(
            index='user_id',
            columns='item_id',
            values='avg_rating'
        ).fillna(0)

        self.R_matrix = df_features.to_numpy()
        self.n_users = int(np.size(self.R_matrix,0))      
        self.n_items = int(np.size(self.R_matrix,1))
        print(np.size(self.R_matrix,0))
        print(np.size(self.R_matrix,1))

        self.muy  = 0
        count = 0
        self.Y = []      # tạo list y chứa các rating != 0 mỗi index có dạng (u,i,value) 
        for u in range(self.n_users):
            for i in range(self.n_items):
                if self.R_matrix[u][i] != 0:
                    self.Y.append(R(u, i, self.R_matrix[u][i]))
                    self.muy = self.muy + self.R_matrix[u][i]
                    count += 1
        self.muy = self.muy / count
        print("Muy:",self.muy)
        self.nfactor = nfactor
        self.anpha = anpha
        self.max_iter = max_iter
        self.lam  = lam

        self.N_Ydata = int(np.size(self.Y))
        
        # temp = self.R_matrix
        # temp.tolist()     
        # self.muy = temp.mean()  # trung bình tất cả Rating trong hệ thống
        # print("Avg muy:",self.muy)

        # Khởi tạo giá trị ban đầu của Ou, Pi
        self.Ou = np.zeros(self.n_users)
        for i in range(self.n_users):
            self.Ou[i] = statistics.mean(self.R_matrix[i,:]) 
        
        self.Pi = np.zeros(self.n_items)
        for i in range(self.n_items):
            self.Pi[i] = statistics.mean(self.R_matrix[:,i])


    def SGD(self):
        print(str(self.n_users) +' : '+str(self.n_items))
        # khời tạo ma trận H V random
        self.H= np.random.uniform(-1, 1, size=(self.n_users, self.nfactor)) 
        self.V= np.random.uniform(-1, 1, size=(self.n_items, self.nfactor)) 
        #lặp với số lần là max_iter
        temp = self.max_iter
        while temp > 0:
            np.random.shuffle(self.Y)
            for i in range(self.N_Ydata):
                r = self.Y[i].rating
                Ou = self.Ou[self.Y[i].user]
                Pi = self.Pi[self.Y[i].item]
                h = self.H[self.Y[i].user,:]
                vT = self.V.transpose()[:,self.Y[i].item]
                v = self.V[self.Y[i].item,:]
                e = r - Ou -Pi - self.muy - h.dot(vT)   # độ lỗi

                H_plus = h - self.anpha*(self.lam*h - e*v)  
                V_plus = v - self.anpha*(self.lam*v - e*h)
                
                self.H[self.Y[i].user,:] = H_plus
                self.V[self.Y[i].item,:] = V_plus
                self.Ou[self.Y[i].user] = Ou +self.anpha*(e - self.lam*Ou)
                self.Pi[self.Y[i].item] = Pi +self.anpha*(e - self.lam*Pi)

            theta = 0
              # ngưỡng lỗi
            for b in range(self.N_Ydata): 
              th = self.H[self.Y[b].user,:]
              tv = self.V.transpose()[:,self.Y[b].item]
              theta =  pow((self.Y[b].rating - self.Ou[self.Y[b].user] - self.Pi[self.Y[b].item] - self.muy - th.dot(tv )), 2)
              if (theta <= pow(10,-6)):
                #print(temp)
                temp = 0
                break
              #print(temp)
              #print(theta, '-', self.Y[b].rating,'-', th.dot(tv))
            temp = temp - 1
            temp -=1
            #print(temp)
        
        # Lưu file
        # H: user factor
        H_pra = pd.DataFrame(self.H, columns = [x + str(y) for x in ['factor'] for y in  range(self.nfactor)] )
        fname = 'FactorMatrix/H.csv'
        H_pra.to_csv(fname, index = None)
        # V: item factor
        V_pra = pd.DataFrame(self.V, columns = [x + str(y) for x in ['factor'] for y in  range(self.nfactor)] )
        fname = 'FactorMatrix/V.csv'
        V_pra.to_csv(fname, index = None)
        Ou_pra = pd.DataFrame(self.Ou, columns =['Bias user'])
        fname = 'FactorMatrix/Ou.csv'
        Ou_pra.to_csv(fname, index = None)
        Pi_pra = pd.DataFrame(self.Pi, columns =['Bias item'])
        fname = 'FactorMatrix/Pi.csv'
        Pi_pra.to_csv(fname, index = None)
    
nfactor = 60
anpha = 0.001
lam = 0.02
max_iter = 500

step2 = Offline('train_data.csv', nfactor, anpha, lam, max_iter)
step2.SGD()
