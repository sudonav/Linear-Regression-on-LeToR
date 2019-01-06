
# coding: utf-8

# In[458]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[431]:


maxAcc = 0.0
maxIter = 0

# Constant for Lambda
C_Lambda = 0.1
#Training Percentage
TrainingPercent = 80
#Validation Percentage
ValidationPercent = 10
#Test Percentage
TestPercent = 10
# Number of clusters
M = 15
PHI = []
IsSynthetic = False


# In[432]:


# Fetches all the values from the given file path and places it in a vector.
# This vector containing the target values are returned.
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

# Fetches all the values from the given file path and places it in a matrix.
# If the data is synthetic some of the features in the data are removed.
# The transpose of the data matrix is calculated and returned.
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)     
    #print ("Data Matrix Generated..")
    return dataMatrix

# Slices 80% of the target values and returns the new vector for training purpose.
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

# Slices 80% of the data values and returns the matrix for training purpose.
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

# The entire dataset is passed as input and 10% of the dataset is returned.
# It is made sure that dataset being returned is only present in either the training, validation, or test data
# based on the training count parameter passed.
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

# The entire target set is passed as input and 10% of the target set is returned.
# It is made sure that target set being returned is only present in either the training, validation, or test target values
# based on the training count parameter passed.
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Generates the Covariance matrix i.e. Big Sigma 41x41 matrix containing the variance in the diagonal elements 
# for the given percentage of the data.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    # Initialize the Big Sigma matrix of zeros
    BigSigma    = np.zeros((len(Data),len(Data)))
    # The entire data is transposed.
    DataT       = np.transpose(Data)
    # Training length is calculated based on the training percent and transposed data.
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    # Variance is calculated for each row in the data and put into the vector.
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    # Diagonal matrix containing the variance of the dataset is generated.
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    # Based on whether the data is synthetic or not, the variance is normalized.
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(0.1,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# Calculates the Radial Basis function
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

# Calculates the Phi(x) for the given data for the given training percent, big sigma and mu matrix.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    print("A:",PHI.shape)
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

# Get the closed form of weights based on the phi matrix, target vector and lambda
# Lambda is an identity matrix of the given lambda value.
# Closed form is given by W* = ((Lambda.I + Phi^T)^-1).Phi^T.t
def GetWeightsClosedForm(PHI, T, Lambda):
    #Lambda matrix is given by Lambda.I
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    # Phi is transposed
    PHI_T       = np.transpose(PHI)
    # (Phi(Transpose).Phi) is calculated
    PHI_SQR     = np.dot(PHI_T,PHI)
    # Lambda matrix is added
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    # Inverse is calculated
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    # The calculated inverse is multiplied with Phi^T
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    # The calculated values is multiplied with T
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    # Return W of size 10
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

# Compute and return Y = W^T.Phi(x)
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# Compute and return the Error root mean square for each of the training, validation and test data
# This gives the accuracy.
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# In[531]:


bigSigma = [1000, 100, 10, 1, 0.1]
trainingAccuracy = [55.27, 55.54, 56.5, 62.6, 64.5]
plt.plot(bigSigma, trainingAccuracy, 'bo-')
plt.xlabel('Big Sigma')
plt.ylabel('Training Accuracy')


# In[533]:


clambda = [1, 0.1, 0.01, 0.001, 0.0001]
trainingAccuracy = [62.4502, 64.5427, 64.5427, 64.5427, 64.5427]
plt.plot(clambda, trainingAccuracy, 'rx-')
plt.xlabel('Lambda')
plt.ylabel('Training Accuracy')


# ## Fetch and Prepare Dataset

# In[433]:


# Get the raw target values i.e. the output labels from the Querylevelnorm_t.csv file into a vector.
RawTarget = GetTargetVector('Querylevelnorm_t.csv')
# Get the raw data i.e. the features and the values that is used to map to an output label
# from the Querylevelnorm_X.csv file which creates a matrix of the entire LeTor data.
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[434]:


# 80% of the entire target values are fetched and placed into TrainingTarget and these targets will be used for training.
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
# 80% of the entire data values are fetched and placed into TrainingData and this data will be used for training.
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[435]:


# 10% of the entire target values are fetched and placed into ValDataAct and these targets will be used for validation.
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
# 10% of the entire data values are fetched and placed into ValData and this data will be used for validation.
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[436]:


# The last 10% of the entire target values are fetched and placed into TestDataAct
# These targets will be used for testing.
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
# The last 10% of the entire data values are fetched and placed into TestData
# These data will be used for testing.
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# In[468]:


TrainingData_T = np.transpose(TrainingData)
distortions = []
K = range(1,20)
for k in K:
    # Clusters the given training data into M clusters i.e. 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(TrainingData_T)
    kmeansModel = kmeans.fit(TrainingData_T)
    distortions.append(sum(np.min(cdist(TrainingData_T, kmeansModel.cluster_centers_, 'euclidean'), axis=1))/TrainingData_T.shape[0])


# In[473]:


fig, ax = plt.subplots() 
ax.plot(K, distortions, 'bx-')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Distortion')
ax.set_title('Optimum number of clusters using Elbow method')
plt.show()


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[ ]:


# Root mean square Error.
ErmsArr = []
# Accuracy
AccuracyArr = []

# Clusters the given training data into M clusters i.e. 10 clusters
kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainingData_T)
# Finds the center of the basis function
Mu = kmeans.cluster_centers_

# Big Sigma is a 41x41 matrix consisting of the covariance of each row in the raw data
# in the diagonal elements and this decides the spread of the basis function.
BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
# This gives the phi value used to find Y in Y = W^T.Phi(x) for the training data
# Phi(x) is linear representation of x in a higher dimensional feature 
# space which is given by the Gaussian Radial Basis function which converts the each 
# vector x into scalar. 
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
# Weights for the closed form is given by W* = ((Lambda.I + Phi^T)^-1).Phi^T.t
# Weights is generated by sum of squares without an regulalizer. Since the matrix is of the form
# MxN we cannot find the inverse directly. Hence we use the Moore - Penrose Pseudo Inverse Matrix
# method to find the closed form solution.
# The form has miminized sum of squared error and an regularizer to avoid overfitting thus 
# This WEIGHT is derived from the closed-form solution with least-squared regularization.
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
# This gives the phi value used to find Y in Y = W^T.Phi(x) for the test data
# Phi(x) is linear representation of x in a higher dimensional feature 
# space which is given by the Gaussian Radial Basis function which converts the each 
# vector of x into scalar. 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
# This gives the phi value used to find Y in Y = W^T.Phi(x) for the validation data
# Phi(x) is linear representation of x in a higher dimensional feature 
# space which is given by the Gaussian Radial Basis function which converts the each 
# vector of x into scalar. 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[515]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[ ]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[534]:


print ('UBITname      = nramanat')
print ('Person Number = 50291712')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = "+str(M)+"\nLambda = "+str(C_Lambda))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[441]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[444]:


W_Now        = np.dot(1, W)
#Lambda
La           = 0.1
#Learning rate
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

# Stochastic gradient descent randomly selects samples of data. After every iteration, the weight is updated
# based on the the learning rate and moves against the gradient of error to reach the global optimum for the
# predicting the output labels.
for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[535]:


print ('----------Gradient Descent Solution--------------------')
print ("M = 15 \nLambda  = 0.1\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

