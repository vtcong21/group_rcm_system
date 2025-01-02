import numpy as np
import pandas as pd
import os
import csv

class AOFRAM_W:
    def __init__(self, G, Beta):
        self.lam = 0.02
        self.Beta = Beta
        self.G = G
        self.n_userGroup = int(np.size(self.G[0])) #! số user trong group
        self.n_Group = int(np.size(self.G, 0)) 
        df_file_csv = pd.read_csv(os.path.join('FactorMatrix/H.csv'))
        self.H = df_file_csv.to_numpy()  

        #! Đọc file V
        df_file_csv = pd.read_csv(os.path.join('FactorMatrix/V.csv'))
        self.Q = df_file_csv.to_numpy()    

        #! Đọc file Ou
        df = pd.read_csv('FactorMatrix/Ou.csv')
        self.Ou = df['Bias user'].values

        #! Đọc file pi
        # df_file_csv = pd.read_csv(os.path.join('Pi.csv'))
        # self.Pi = df_file_csv.to_numpy()
        df = pd.read_csv('FactorMatrix/Pi.csv')
        self.Pi = df['Bias item'].values

        #! Đọc Influ = Ku
        # df_file_csv = pd.read_csv(os.path.join('Influ.csv'))
        # self.Influ = df_file_csv.to_numpy()
        df = pd.read_csv('FactorMatrix/Influ.csv')
        self.Influ = df['Influ'].values

        # chuyển file Rating thành ma trận
        df_ratings = pd.read_csv(
            os.path.join("train_data.csv"),
            usecols=['user_id', 'item_id', 'avg_rating'],
            dtype={'user_id': 'string', 'item_id': 'string', 'avg_rating': 'float32'})

        # df_copy = df_ratings.copy()

        # user_id_mapping = {user_id: i+1 for i, user_id in enumerate(df_copy['user_id'].unique())}
        # item_id_mapping = {item_id: i+1 for i, item_id in enumerate(df_copy['item_id'].unique())}

        # df_copy['user_id'] = df_copy['user_id'].map(user_id_mapping)
        # df_copy['item_id'] = df_copy['item_id'].map(item_id_mapping)

        df_features = df_ratings.pivot_table(
            index='user_id',
            columns='item_id',
            values='avg_rating'
        ).fillna(0)
        self.R_matrix = df_features.to_numpy()

        self.n_users = int(np.size(self.R_matrix, 0)) 
        self.n_items = int(np.size(self.R_matrix, 1))
        print(self.n_users,"-",self.n_items)
        arr = np.array(self.R_matrix)

        self.G_RLM = np.zeros((self.n_Group, self.n_items)) # rating at least one member
        self.G_RAM = np.zeros((self.n_Group, self.n_items)) # rating all member
        # Caculate the Avg of all observed rating
        self.Muy  = 0
        count = 0
        self.Y = []      # tạo list y chứa các rating != 0 mỗi index có dạng (u,i,value) 
        for u in range(self.n_users):
            for i in range(self.n_items):
                if self.R_matrix[u][i] != 0:
                    self.Muy = self.Muy + self.R_matrix[u][i]
                    count += 1
        self.Muy = self.Muy / count
        #self.Muy = arr.mean()



    def Caculate_Predict_Rating(self,user,item):
        return self.Ou[user]+self.Pi[item]+self.Muy+np.dot(np.array(self.H[user]),np.array(self.Q[item]))
    
    def Caculate_Ku_Beta(self):
        print("---> Ku_Beta\n")
        List_Influ = []
        for j in range(self.n_Group):
            temp = []
            for i in range(self.n_userGroup):
                temp.append(self.Influ[G[j][i]-1])
            List_Influ.append(temp)

        self.List_Ku_Beta = []
        for j in range(self.n_Group):
            l=[]
            for u in range(self.n_userGroup):
                x = max(List_Influ[j])-min(List_Influ[j])
                if(x == 0):
                    a = 0
                else:
                    a = (List_Influ[j][u]-min(List_Influ[j]))/x
                l.append(self.Beta+(1.0-self.Beta)*a)
            self.List_Ku_Beta.append(l)


        dffile = pd.DataFrame(self.List_Ku_Beta, columns = [x + str(y) for x in ['User'] for y in  range(self.n_userGroup)] )
        fname = './Profile/' + 'PF_Ku_Beta_G'+ str(self.n_userGroup)+'.csv'
        dffile.to_csv(fname, index = None)

                
    def Caculate_Cui_Beta(self):
        fname = './Profile/' + 'PF_Cui_Beta_G'+ str(self.n_userGroup)+'.csv'
        if os.path.exists(fname):
            os.remove(fname)
        dffile = pd.DataFrame([], columns = [x + str(y) for x in ['Item'] for y in  range(self.n_items)])
        dffile.to_csv(fname, index = None)

        fname_group = './Group/Group' + str(self.n_userGroup) + '.csv'
        df_Group = pd.read_csv(fname_group)
        G_matrix = df_Group.to_numpy()

        for g in range(self.n_Group):
            temp = []
            for u in range(self.n_userGroup):
                l = []
                for i in range(self.n_items):
                    if self.R_matrix[G[g][u]-1][i] != 0:
                        c = abs(self.R_matrix[G[g][u]-1][i]-self.Caculate_Predict_Rating(G[g][u]-1,i))
                    else:
                        c = abs(self.Ou[G[g][u]-1]+self.Pi[i]+self.Muy-np.dot(np.array(self.H[G[g][u]-1]),np.array(self.Q[i])))
                    l.append(c)
                for i in range(self.n_items):
                    l[i] = (l[i] - min(l))/(max(l)-min(l))
                

                    #l.append(1-c)
                temp.append(l)

            with open(fname, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Ghi từng hàng trong danh sách data vào tệp CSV
                for row in temp:
                    writer.writerow(row)

        df = pd.read_csv(fname)
        line_count = len(df)
        row_count = len(df.columns)

    def Caculate_Weights(self):
        self.Weight = []
        Cui_fname = './Profile/' + 'PF_Cui_Beta_G'+ str(self.n_userGroup)+'.csv'
        Ku_fname = './Profile/' + 'PF_Ku_Beta_G'+ str(self.n_userGroup)+'.csv'
        df_Ku = pd.read_csv(Ku_fname)
        Ku_Matrix = df_Ku.to_numpy()
        df_Cui = pd.read_csv(Cui_fname)
        Cui_Matrix = df_Cui.to_numpy()

        if os.path.exists('./Profile/PF_Weight.csv'):
            os.remove('./Profile/PF_Weight.csv')
        dffile = pd.DataFrame([], columns = [x + str(y) for x in ['Item'] for y in  range(self.n_items)])
        fname = './Profile/PF_Weight.csv'
        dffile.to_csv(fname, index = None)
        for g in range(self.n_Group):
            temp = []
            for u in range(self.n_userGroup):
                l = []
                for i in range(self.n_items):
                    w = Ku_Matrix[g][u] * Cui_Matrix[g*self.n_userGroup+u][i]
                    l.append(w)
                temp.append(l)
                self.Weight.append(l)
            with open('./Profile/PF_Weight.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Ghi từng hàng trong danh sách data vào tệp CSV
                for row in temp:
                    writer.writerow(row)


    def RLM(self):    
        print('Tới RLM rồi nè!')
        print("Ma trận Weight có ", str(np.size(self.Weight, 0)), "dòng !")
        for a in range (self.n_Group):
            for b in range (self.n_items):
                tu = 0.0
                mau = 0.0
                for c in range (self.n_userGroup):
                    user = self.G[a,c]      #! lấy index thứ c tại group a 
                    rating = self.R_matrix[user-1, b]   #! lấy rating cúa user cá nhân
                    infl = self.Influ[user-1]
                    tu += rating*self.Weight[a*self.n_userGroup+c][b]
                    if rating != 0:
                        mau += infl
                if(mau !=0):
                    self.G_RLM[a,b] = tu/mau
                else:
                    self.G_RLM[a,b] = 0
    def RAM(self):
        Weight_fname = './Profile/PF_Weight.csv'
        df_Weight = pd.read_csv(Weight_fname)
        self.Weight = df_Weight.to_numpy()
        print('Tới RAM rồi nè!')
        for i in range (self.n_items):
            for g in range(self.n_Group):
                check = False
                for u in range (self.n_userGroup):
                    userid = self.G[g][u]
                    if(self.R_matrix[userid-1][i] != 0):
                        check = True
                
                if(check):
                    sum_avg = 0.0
                    sum_weight = 0.0
                    for c in range (self.n_userGroup):
                        userid = self.G[g,c]      #! lấy index thứ c tại group a
                        if(self.R_matrix[userid-1][i] == 0):
                            rating = self.Caculate_Predict_Rating(userid-1,i)
                        else:  
                            rating = self.R_matrix[userid-1, i]   #! lấy rating cúa user cá nhân

                        sum_avg += rating*self.Weight[g*self.n_userGroup+c][i]
                        sum_weight += self.Weight[g*self.n_userGroup+c][i]
                    if(sum_weight != 0):
                        self.G_RAM[g,i] = sum_avg/sum_weight
                    else:
                        self.G_RAM[g,i] = 0



    def Ridge_regression(self, type): 
        print('Tới Ridge_regression rồi nè!')
        # Class này dùng để tính Hg và Og
        self.type = type
        self.col_factor = int(np.size(self.Q, 1))
        self.Og = np.zeros( self.n_Group)
        self.Hg = np.zeros((self.n_Group, self.col_factor))
        if (self.type == 'RLM'):
            self.GM = self.G_RLM
        elif (self.type == 'RAM'):
            self.GM = self.G_RAM
        for a in range(self.n_Group):
            temp = list(self.GM[a,:])
            available_item = self.n_items - int(temp.count(0)) # số lượng item mà G có đánh giá
            # tao ma trạn 2 chieu _V chỉ chứa các vector item mà _g có đánh giá
            self._V = np.zeros((available_item,self.col_factor))
            # tao vector _pi, _g
            self._pi= np.zeros((1,available_item))
            # tao vector co gia tri 1
            z = np.ones((1,available_item))
            self._g =[]
            i = 0
            for b in range(self.n_items):
                rating = self.GM[a,b]
                if rating !=0:
                    self._g.append(rating)
                    self._V[i,:] = self.Q[b,:]
                    self._pi[0,i] = self.Pi[b]
                    i +=1
            I = np.eye(self.col_factor + 1)
            _V_col_end = np.ones((available_item,1))
            # thêm một cột có giá trị 1 ở cuối ma trận
            self._V = np.hstack((self._V,_V_col_end))
            # Tính công thức
            temp1 = (self._g - self._pi - self.Muy*z).dot(self._V)
            temp2 = np.linalg.pinv( self._V.transpose().dot(self._V) + self.lam*I )
            hg = temp1.dot(temp2)
            
            # (self._g - self._pi - self.muy*z)*self._V*(self._V.transpose().dot(self._V) + self.lam*I)^-1
            # Tách kết quả, phần tử cuối là og, còn lại là của hg
            og = hg[0,self.col_factor]
            hg = np.delete(hg, np.s_[-1])
            self.Og[a] = og
            self.Hg[a, :] = hg
        #! Kết quả cuối cùng của Ridge Regression là Group Factor Hg và bias group Og

    def Find_ListItem(self):
        print('Tới Find List rồi nè!')
        print('Tới list')
        self.L_item = []
        for a in range (self.n_Group): 
          temp = []
          for b in range (self.n_items): 
            tt = 0
            for c in range (self.n_userGroup):
              user = self.G[a, c] 
              if (self.R_matrix[user-1, b] == 0):     #! lấy các item chưa được user nào trong Group trải nghiệm
                tt +=1
            if (tt == self.n_userGroup):
              temp.append(b)
          self.L_item.append(temp)     #! mỗi item trong L là 1 list item khuyết

    def Find_ratingG(self):
        print('Tới Find RG rồi nè!')
        self.List_ItemG = []
        count = 0
        sum1 = 0
        for a in range(self.n_Group):
          temp = []
          for b in (self.L_item[a]): #! xét mỗi item trong list vừa tìm đc
            ratingG = self.Og[a] + self.Pi[b] + np.dot(np.array(self.Hg[a]),np.array(self.Q[b])) + self.Muy
            sum1 += ratingG
            count+=1
            if (ratingG >= 3):
              temp.append(b)
          self.List_ItemG.append(temp)
        dffile =  pd.DataFrame(self.List_ItemG)
        fname = './Profile/PF_AVG_G'+ str(self.n_userGroup)+'.csv' #! => Mỗi dòng trong file là 1 Group và
                                                                    #! danh sách các item đề xuất cho Group đó
        dffile.to_csv(fname, index = None)
    #! --> Tính rating của các group cho các item khuyết

    def Profile_Aggregation(self):
        df_Weight = pd.read_csv('./Profile/PF_Weight.csv')
        Weight_Matrix = df_Weight.to_numpy()

        df_Group = pd.read_csv('./Group/Group3.csv')
        Group = df_Group.to_numpy()
        self.Profile_Aggregation = []
        for g in range(self.n_Group):
            l = np.zeros(self.n_items).tolist()
            sum_weight = np.zeros(self.n_items).tolist()
            for u in range(self.n_userGroup):
                for i in range(self.n_items):
                    if(self.R_matrix[G[g][u] - 1][i] != 0):
                        l[i] += self.R_matrix[G[g][u] - 1][i] * Weight_Matrix[g*self.n_userGroup+u][i]
                    else:
                        l[i] += self.Caculate_Predict_Rating(G[g][u] - 1, i) * Weight_Matrix[g*self.n_userGroup+u][i]
                    sum_weight[i] += Weight_Matrix[g*self.n_userGroup+u][i]

            for idx in range(self.n_items):
                l[idx] = l[idx] / sum_weight[idx]
            for idx in range(self.n_items):
                l[idx] = 5*(l[idx] - min(l))/(max(l)-min(l))
            
            self.Profile_Aggregation.append(l)

        dffile = pd.DataFrame(self.Profile_Aggregation, columns = [x + str(y) for x in ['Item'] for y in  range(self.n_items)])
        fname = './Profile/PF_Profile_Aggregation.csv'
        dffile.to_csv(fname, index = None)

        df = pd.read_csv('./Profile/PF_Profile_Aggregation.csv')
        line_count = len(df)
        row_count = len(df.columns)
        print(f'Số dòng trong tệp CSV: {line_count}')
        print(f'Số cột trong tệp CSV: {row_count}')


nUserGroup = 4
file_group = './Group/Group' + str(nUserGroup) + '.csv'
if(nUserGroup == 2):
    df_group = pd.read_csv(
        os.path.join(file_group),
        usecols=['User0','User1'],
        dtype={'User0': 'int32' ,'User1': 'int32'})
elif nUserGroup == 3:
    df_group = pd.read_csv(
        os.path.join(file_group),
        usecols=['User0','User1','User2'],
        dtype={'User0': 'int32' ,'User1': 'int32','User2':'int32'})
elif nUserGroup == 4:
    df_group = pd.read_csv(
        os.path.join(file_group),
        usecols=['User0','User1','User2','User3'],
        dtype={'User0': 'int32' ,'User1': 'int32','User2':'int32','User3':'int32'})
elif nUserGroup == 5:
    df_group = pd.read_csv(
        os.path.join(file_group),
        usecols=['User0','User1','User2','User3','User4'],
        dtype={'User0': 'int32' ,'User1': 'int32','User2':'int32','User3':'int32','User4':'int32'})

G = df_group.to_numpy() 
step = AOFRAM_W(G,0.4)
#step.CheckPredict()
step.Caculate_Ku_Beta()
step.Caculate_Cui_Beta()
step.Caculate_Weights()
step.RAM()
step.Ridge_regression('RAM')
step.Find_ListItem()
step.Find_ratingG()
#step.Profile_Aggregation()