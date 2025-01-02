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

class Measure:
    def __init__(self, Testing, G, IRG, fname):
        self.test = Testing         # test matrix
        self.n_items = int(np.size(self.test,1))
        self.n_users = int(np.size(self.test,0))
        self.IRG = IRG
        self.n_Group = len(self.IRG)
        self.G = G
        self.n_userGroup = int(np.size(self.G,1))
        self.fscore = np.zeros((self.n_Group, 6)) 
        self.fname = fname

        if(self.n_userGroup % 2 == 0):
            self.Standard = int(self.n_userGroup / 2)
        else:
            self.Standard = int(self.n_userGroup / 2) + 1
        
        print(self.Standard)

        print(self.n_users," users and ", self.n_items," item\n")
    def Fscore(self):
            print('Vô   F-score r nè!')
            self.T = []
            self.C = []
            # g là groupid
            for g in range(self.n_Group):
                tempT = []
                tempC = []
                if (len(self.IRG[g])==0): #nếu group đó ko có item nào đk đề xuất
                    self.T.append(tempT)
                    continue
                # b là mỗi item được recommend cho group
                
                for d in self.IRG[g]:   #! mỗi index IRG là 1 list các item đề xuất cho Group đó
                    # c là số phần tử (user) trong 1 group
                    #  temp2 là biến điều kiện
                    
                    b= int(float(d))    #! index của item
                    score = 0
                    count = 0
                    for c in range(self.n_userGroup):
                        userid =  self.G[g,c]
                        if(userid >= self.n_users or b >= self.n_items or self.test[userid-1,b] == 0):
                            count += 1
                        elif (self.test[userid-1,b] >= 3):
                            score += 1
                    #! item phải đc tất ít nhất nửa thành viên đánh giá > 3 mới đc tính
                    if(self.n_userGroup % 2 == 0):
                        if (count <= self.Standard):
                            tempC.append(b)
                        if (count <= self.Standard and score >= self.Standard):
                            tempT.append(b)
                    elif (self.n_userGroup == 3):
                        if (count <= 1):
                            tempC.append(b)
                        if (count <= 1 and score >=2):
                            tempT.append(b)
                    else:
                        if (count < self.Standard):
                            tempC.append(b)
                        if (count < self.Standard and score >= self.Standard):
                            tempT.append(b)
                self.T.append(tempT)
                self.C.append(tempC)
                #! C là danh sách item dự đoán của hệ thống (kết quả của giai đoạn gợi ý)
                #! T là danh sách tư vấn chính xác trên thực tế

            print('Xong T!')
            for g in range (self.n_Group):
                if (len(self.T[g]) > 1):
                    precisionG = len(set(self.T[g]) & set(self.C[g])) / len(self.C[g])
                    recallG = len(set(self.T[g]) & set(self.C[g])) / len(self.T[g])
                    if (precisionG ==0 and recallG ==0):
                        fscoreG = 0
                    else: 
                        fscoreG = 2*precisionG*recallG/(precisionG + recallG)
                        self.fscore[g,0] = precisionG
                        self.fscore[g,1] = recallG
                        self.fscore[g,2] = fscoreG
                        self.fscore[g,3] = len(self.T[g])
                        self.fscore[g,4] = len(self.C[g])
                self.fscore[g,5] = len(self.IRG[g])

            print("Length:",len(self.IRG[0]))
            print("Length:",len(self.IRG[1]))
            print('Xong fscore sắp lưu')
            t1 = pd.DataFrame(self.G, columns = [x + str(y) for x in ['User '] for y in  range(self.n_userGroup)])
            
            t3 = pd.DataFrame(self.fscore, columns = ['precisionG', 'recallG', 'fscoreG','T','C','Quantity'])
            t4 = pd.concat([t1,t3], axis = 1)
            t4.to_csv(self.fname)


# read test file
df_testing = pd.read_csv(
    os.path.join("test_data.csv"),
    usecols=['user_id', 'item_id', 'avg_rating'],
    dtype={'user_id': 'int32', 'item_id': 'int32', 'avg_rating': 'float32'})
# init matrix fill 0
df_testing_features = df_testing.pivot_table(
    index='user_id',
    columns='item_id',
    values='avg_rating'
).fillna(0)

testing = df_testing_features.to_numpy()


# Đọc file Group 
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

G2 = df_group.to_numpy() 
# Đọc danh sách gợi ý cho G
ffname = './Profile/PF_AVG_G'+str(nUserGroup) + '.csv'
lIG =[]
with open(ffname) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        bbb=[]
        if line_count == 0:
            line_count = 1
            continue
        else:
            bbb =  [x for x in row[:] if x != '']
        lIG.append(list(bbb)) 


nUserGroup = np.size(G2[0])
fname = './Measure/Fscore_G' + str(nUserGroup)+  '.csv'
step4 = Measure(testing, G2, lIG, fname)
step4.Fscore()

