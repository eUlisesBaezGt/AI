#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# we use the tool to can import documentas from drive to google colab 
# from google.colab import drive 
# drive.mount('/content/drive')


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from copy import copy 
from scipy import stats
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


#Here we import the stock price document from drive 
s_p_df = pd.read_csv('stock.csv')
s_p_df


# In[5]:


#Here we import the stock volumen document from drive 
s_vol_df =pd.read_csv('stock_volume.csv')
s_vol_df


# In[6]:


#We reorginize the date from the stok price chart 
s_p_df = s_p_df.sort_values(by = ['Date'])
s_p_df


# In[7]:


#We reorginize the date from the stok volume chart 
s_vol_df = s_vol_df.sort_values(by = ['Date'])
s_vol_df


# In[8]:


#Make the search of nnon exist data in satok price
s_p_df.isnull().sum()


# In[9]:


#Make the search of nnon exist data in stock volume
s_vol_df.isnull().sum()


# In[10]:


# We need to normalize the stok prices 
# "The normalized value is calculated by dividing the stock price by its moving average"
# for more info of this    https://itadviser.dev/stock-market-data-normalization-for-time-series/#:~:text=Moving%20average%20normalization%20smoothens%20out,price%20by%20its%20moving%20average.

def nom(df):
# Here is copy the char at ones x move for actulize the old info 
  x = df.copy()
# Take the colums of prices exept the first one because thats the date colum and names of the enterprice 
  for i in x.columns[1:]:
# Here make the operation to normalize 
    x[i] = x[i]/x[i][0]
  return x


# In[11]:


# I use interactive ploit to see the prices in a interactive chart 
def inter_plot(df, title):
# we use px to creat entire figure at ones 
# https://plotly.com/python/plotly-express/
  fig = px.line(title = title)
# Make a for loop to create every image for every price and dates 
  for i in df.columns[1:]:
# star to make scaters for every info for every enterprice
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()


# In[12]:


# Visualize the chart of the stock prices
inter_plot(s_p_df,'STOCK PRICE') 


# In[13]:


# Visualize the chart of the stock volume
inter_plot(s_vol_df,'STOCK VOLUME') 


# In[14]:


# Visualize the chart of the normalize stock price
inter_plot(nom(s_p_df),'NORMALIZED STOCK PRICE') 


# In[15]:


# Visualize the chart of the normalize stock volume
inter_plot(nom(s_vol_df),'NORMALIZED STOCK VOLUME') 


# In[16]:


#mke a function to convine date, stok price, volume in onde df 
def in_info(pri_df,vol_df,name):
#Make a dictionary to bring the information fast and organize 
  return pd.DataFrame({'Date':pri_df['Date'], 'Close':pri_df[name], 'Volume':vol_df[name]})


# In[17]:


#Create a function to the target for the IA Model 
def traiding(data):
  n = 1
#create a new column that shifts one day to make the model for trainig 
  data['Target'] = data[['Close']].shift(-n)
  return data


# In[18]:


#test the model with S&P500
p_vol_df = in_info(s_p_df, s_vol_df, 'TSLA')
p_vol_df


# In[19]:


# Test "trading"
p_vol_tar_df = traiding(p_vol_df)
p_vol_tar_df


# In[20]:


#Remove the last row decause it was null
p_vol_tar_df = p_vol_tar_df[:-1]
p_vol_tar_df


# In[22]:


# Scale the data and proces the data 
# specify the object of the class
sc = MinMaxScaler(feature_range = (0,1))
# Aplay the transformation to the object and normalize the colums exept date
p_vol_tar_sca_df = sc.fit_transform(p_vol_tar_df.drop(columns = ['Date']))
p_vol_tar_sca_df


# In[23]:


# Create the imput and the ouput
#input
# we tell that we need all the row and the 2 first columns
x = p_vol_tar_sca_df[:, :2]
# ouput 
# we only take the final column and all the rows 
y = p_vol_tar_sca_df[:, 2:]


# In[24]:


#Chake the data 
x , y, y.shape, x.shape


# In[25]:


#Spit the date for the train and the other part for the prediction 
spl = int(0.65 * len(x))
spl


# In[26]:


#give the date to training to x 
x_trn = x[:spl]
x_trn.shape


# In[27]:


#give the date to training to y
y_trn = y[:spl]
y_trn.shape


# In[28]:


# give the test data 
x_tst = x[spl:]
y_tst = y[spl:]


# In[29]:


# Check the test 
x_tst.shape, y_tst.shape


# In[30]:


# define the predict function 
def s_plot(data,title):
# give the axes dimentions
  plt.figure(figsize = (13, 5))
#give the data to the chart 
  plt.plot(data, linewidth = 3)
# Put a title
  plt.title(title)
#add a grid 
  plt.grid()

s_plot(x_trn, 'TRAINING DATA')
s_plot(x_tst, 'TEST DATA')


# In[ ]:


# SImple linear regresion model 
# The simple regression model assumes a linear relationship, Y = α + βX + ε, 
# between a dependent variable Y and an explanatory variable X, with the error 
# term ε encompassing omitted factors
# more info 
# https://www.sciencedirect.com/topics/mathematics/simple-regression-model
# Easy formula y = b + m * x
# y = dependent variable 
# b = model 
# m = goal 
# x = independent variable


# In[31]:


# Creating the Ridge trianing for the linear regresion model 
#Ridge solves the linear regresion model problem 
reg_mode = Ridge()
reg_mode.fit(x_trn, y_trn)


# In[32]:


#Testing the model 
#calculate the pression 
lr_prs = reg_mode.score(x_tst, y_tst)
print('RIDGE REGRESSION SCORE:', lr_prs)


# In[33]:


#Make prediction 
prd_prc = reg_mode.predict(x)
prd_prc


# In[34]:


# Put the predicted values in a list 
prd = []
for i in prd_prc:
  prd.append(i[0])

#check the lend 
len(prd)


# In[35]:


# put the close values into the list 
cls = []
for i in p_vol_tar_sca_df:
  cls.append(i[0])

#Check the lend 
len(cls)


# In[36]:


#make a df for indivial stock data 
df_prd = p_vol_tar_df[['Date']]
df_prd


# In[37]:


#ad the close to the df 
df_prd['Close'] = cls
df_prd


# In[38]:


# ad the predicted values to the df 
df_prd ['Prediction'] = prd
df_prd


# In[39]:


#Make the grafic 
inter_plot(df_prd, 'Original VS Predictions ')


# #Train a LSTM Time series model 

# In[40]:


#Make the test
p_vol_df = in_info(s_p_df, s_vol_df, 'TSLA')
p_vol_df


# In[41]:


# get the input data 
trn_data = p_vol_df.iloc[:,1:3].values
trn_data


# In[42]:


# NOrmalize data 
sc = MinMaxScaler(feature_range = (0, 1))
trn_s_scl = sc.fit_transform(trn_data)



# In[43]:


#CReate the training and testing 
x = []
y = []

for i in range(1, len(p_vol_df)):
  x.append(trn_s_scl[i-1:i, 0])
  y.append(trn_s_scl[i, 0])


# In[44]:


# Check x and y 
x, y


# In[45]:


# convert to array format 
x = np.asarray(x)
y = np.asarray(y)


# In[46]:


#Check x and y 
x, y 


# In[47]:


#split data for the training and prediction 
split = int(0.7 * len(x))
x_trn = x[:split]
y_trn = y[:split]
x_tst = x[split:]
y_tst = y[split:]


# In[48]:


#Make the 1D array to 3D array 
x_trn = np.reshape(x_trn, (x_trn.shape[0], x_trn.shape[1],1))
x_tst = np.reshape(x_tst, (x_tst.shape[0], x_tst.shape[1],1))
x_trn.shape, x_tst.shape


# In[49]:


x_trn.shape[2]


# In[66]:


# Feed the model 
inputs = keras.layers.Input(shape = (x_trn.shape[1], x_trn.shape[2]))

x = keras.layers.LSTM(400, return_sequences = True)(inputs)
x = keras.layers.LSTM(400, return_sequences = True)(x)
x = keras.layers.LSTM(400, return_sequences = True)(x)
outputs = keras.layers.Dense(1, activation = 'linear')(x)

model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()


# In[56]:


#Training the model 
hst = model.fit(x_trn, y_trn, epochs = 50, batch_size = 32, validation_split = 0.2)


# In[80]:


x
# <KerasTensor: shape=(None, 1, 400) dtype=float32 (created by layer 'lstm_11')>
# We need None to be a Number to use it in predictions cell below


# In[85]:


#Limpiez de datos necesrio 
# Make predictions


# In[84]:


# Appedn predictes
tst_prd = []
for i in predictions:
  tst_prd.append(i[0]) 


# In[61]:


#Testing predictions 
tst_prd


# In[62]:


# Make a df for the prediction 
df_prd = p_vol_df[1:][['Date', 'Close']]
df_prd


# In[63]:


# ad a column for the predictions
df_prd['Predictions'] = tst_prd


# In[ ]:


# test the df 
df_prd


# In[ ]:


close = []
for i in trn_s_scl:
  close.append(i[0])


# In[ ]:


#testinf close values 
close 


# In[ ]:


# make all ones 
df_prd['Close'] = close[1:] 


# In[ ]:


# Test the new table 
df_prd


# In[ ]:


#Make the plot frame 
inter_plot(df_prd, 'ORIGINAL PRIVE VS PREDICTES LSTM')

