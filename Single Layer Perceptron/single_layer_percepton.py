import pandas as pd    
import matplotlib.pyplot as plt  
import math as m

#dataset import
idx = ['x1','x2','x3','x4','name']
df=pd.read_csv('E:\iris.csv',names=idx)

#representate type or species into binary representation 
df=df.assign(code=0.0)

for i,j in enumerate(df.name):
    df.code[i] = 1.0 if j == 'setosa' else 0.0

#convert dataset into list/matrix
dataset = df.head(100).values.tolist()

#divide dataset into 5 section
section1 = dataset[:10]+dataset[90:100]
section2 = dataset[10:20]+dataset[80:90]
section3 = dataset[20:30]+dataset[70:80]
section4 = dataset[30:40]+dataset[60:70]
section5 = dataset[40:50]+dataset[70:80]

#define training data
train1 = section1[:]+section2[:]+section3[:]+section4[:]
train2 = section2[:]+section3[:]+section4[:]+section5[:]
train3 = section1[:]+section3[:]+section4[:]+section5[:]
train4 = section1[:]+section2[:]+section4[:]+section5[:]
train5 = section1[:]+section2[:]+section3[:]+section5[:]

train=[train1,train2, train3,train4, train5]
validasi= [section5, section1, section2, section3, section4]

#initiate data weigth
weigth = [0.5,0.5,0.5,0.5]
listWeigth = [weigth[:] for i in range(5)]
dWeigth = [0.5,0.5,0.5,0.5]
bias = 0.5
listBias=[bias for i in range (5)]
dbias = 0

error_train_final = []
error_val_final = []

acc_train_final = []
acc_val_final = []


#summary function
def total(x,j):
    totalWeigth=0
    global listWeigth
    for i in range (4):
      totalWeigth+= (x[i]*listWeigth[j][i])
    
    global bias
    totalWeigth=totalWeigth+bias
    
    return totalWeigth

 #sigmoid function 
def activation(totalWeight):
    aktivasi= (1/(1+(m.exp(-totalWeight))))
    return aktivasi

#derivative neuron weigth
def dt(x, target,act):
    global dWeigth
    for i in range (4):
       dWeigth[i]= 2*x[i]*act*(act-target)*(1-act)
#derivative bias neuron weigth      
def dtBias(target,act):
    global dbias
    dbias=2*bias*act*(target-act)*(1-act)

#calculate error
def error(act,target):
    error = pow((target-act),2)
    return error
 
#representation prediction
def prediction(act):
    if (act>=0.5):
        return 1
    else:
        return 0

#updating weigth
def newWeigth(j,lRate):
    global listWeigth
    for i in range(4):
      listWeigth[j][i]= listWeigth[j][i]-(lRate*dWeigth[i])
      
#updating bias
def newBias(j,lRate):
    global listBias
    listBias[j]= listBias[j]- (lRate*dbias)

#main program
def main(lR):
    for i in range (300):
      sumErr_Train=0.0
      sumErr_Val=0.0
      sumAcc_Train=0.0
      sumAcc_Val=0.0
      
      for j in range (5):
        sum1=0
        sum2=0
        tptn = 0
        tptn2 = 0
        
        #training
        for k in range (80):
          act= activation(total(train[j][k],j))
          pred= prediction(act)
          if ( pred== train[j][k][5]):
            tptn+=1
            
           
          sum1+= error(train[j][k][5],act)
         
          dt(train[j][k][0:4],train[j][k][5],act)
          dtBias(train[j][k][5],act)
    
          newWeigth(j,lR)
          newBias(j,lR)
          
        sumErr_Train += sum1/80 
        sumAcc_Train += (tptn/80)*100
          
          
        #validasi
        for l in range (20):
          acti= activation(total(validasi[j][l],j))
          predict = prediction(acti)
          if ( predict== validasi[j][l][5]):
            tptn2+=1
            
          sum2+= error(validasi[j][l][5],acti)
        sumErr_Val += sum2/20 
        sumAcc_Val += (tptn2/20)*100
    
      error_train_final.append(m.log(sumErr_Train/5))
      error_val_final.append(m.log(sumErr_Val/5))
      acc_train_final.append(m.log(sumAcc_Train/5))
      acc_val_final.append(m.log(sumAcc_Val/5))
      plt.figure(1)
      
    plt.plot(acc_train_final,'r-', label='train')
    plt.plot(acc_val_final,'y-', label='validasi')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    
    plt.figure(2)
    plt.plot(error_train_final,'r-', label='training')
    plt.plot(error_val_final,'y-', label='validasi')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(loc='upper left')
    plt.show()

#learing rate =0.1
#main(0.1)

#learing rate=0.8
main(0.8)

