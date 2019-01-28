import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Loading the data through file
df =pd.read_csv('smspam',sep='\t',names=['status','message'])  #Loading the data through file

#changing the sting spam and ham with '1' and '0'
df.loc[df['status']=='ham','status']=1        
df.loc[df['status']=='spam','status']=0

#df_x is input features
df_x=df['message']
#df_y is  label for given training set
df_y=df['status']

#Creating our CountVectoriser for bag of words model
cv=TfidfVectorizer()

#splitting our data into training set and text set
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2)

#cv.fit_transform method is used to create our dictionary based on training set and it returns an vector with how many times a word is repeated
x_train=cv.fit_transform(x_train).toarray()
#cv.transform method transform the given text into vector
x_test=cv.transform(x_test).toarray()


#Creating our Classifier
mnb=MultinomialNB()

#as it is a classification problem we need to convert our Labels into 'int'
y_train=y_train.astype('int')
y_test= y_test.astype('int')

#fittting our data in model
mnb.fit(x_train,y_train)
#predicting the result for test set
pred =mnb.predict(x_test)

#Checking the accuracy of model
score= accuracy_score(y_test, pred)

print(score)
