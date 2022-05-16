import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime as dt
# %matplotlib inline
# path = "R:\DYP\8\LP 3\h.csv"
dataset = pd.read_csv('Linear Regression\hours.csv')


#dataset=pd.read_csv(&quot;h.csv&quot;)
X=dataset.iloc[:,:-1].values
print(X)
y=dataset.iloc[:,1].values
print(y)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
print("Accuracy : ", str(regressor.score(X, y) * 100))
y_pred=regressor.predict([[10]])
print(y_pred)



hours=int(input("Enter the no of hours:"))
eq=regressor.coef_*hours+regressor.intercept_
print("y = %f*%f+%f&" %(regressor.coef_,hours,regressor.intercept_))
print("Equation for best fit line is: y=%f*x+%f" %(regressor.coef_,regressor.intercept_))
print("Risk Score: ", eq[0])



plt.plot(X, y, 'o')
plt.plot(X, regressor.predict(X));
plt.show()


df=pd.DataFrame({'X':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],
'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
f1 = df['X'].values
f2 = df['y'].values
X = np.array(list(zip(f1, f2)))
print(X)
C_x=np.array([0.1,0.6])
C_y=np.array([0.3,0.2])
centroids=C_x,C_y
colmap = {1: 'r', 2: 'b'}
plt.scatter(f1, f2, color='k')
plt.show()


plt.scatter(C_x[0],C_y[0], color=colmap[1])
plt.scatter(C_x[1],C_y[1], color=colmap[2])
plt.show()


C = np.array(list((C_x, C_y)), dtype=np.float32)
print (C)
plt.scatter(f1, f2, c='#050505')
plt.scatter(C_x[0], C_y[0], marker='*', s=200, c='r')
plt.scatter(C_x[1], C_y[1], marker='*', s=200, c='b')
plt.show()



from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
model.fit(X)
labels=model.labels_
print(labels)
count=0
for i in range(len(labels)):
    if (labels[i]==1):
        count=count+1
print('No of population around cluster 2:',count-1)




new_centroids = model.cluster_centers_
print('Previous value of m1 and m2 is:')
print('M1==',centroids[0])
print('M1==',centroids[1])
print('updated value of m1 and m2 is:')
print('M1==',new_centroids[0])
print('M1==',new_centroids[1])