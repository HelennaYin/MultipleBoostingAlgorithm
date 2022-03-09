## Multiple Boosting Algorithm
```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
```
First we need to build the base model. Here I chose locally weighted regression. 
```
def Tricubic(x):
    if len(x.shape) == 1:
      x = x.reshape(-1,1)

    d = np.sqrt(np.sum(x**2,axis=1))
    return np.where(d>1,0,70/81*(1-d**3)**3)
```

```
def lw_reg(X, y, xnew, kern, tau, intercept):
    n = len(X)
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)

    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)])

    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) 
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

```
def booster(X,y,xnew,kern,tau,model_boosting,intercept,nboost):
  Fx = lw_reg(X,y,X,kern,tau,intercept) 
  Fx_new = lw_reg(X,y,xnew,kern,tau,intercept)
  new_y = y - Fx #that is our residual after we fit the first locally weighted regression model
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
```

```
data = pd.read_csv('...path/concrete.csv')
y = data['strength'].values
X=data[['cement', 'water', 'coarseagg']].values
```






## LightGBM
LightGBM stands for light gradient boosting. This algorithm has multiple advantages. LightGBM has been proven more accurate and less time-consuming in model training compared to other boosting algorithm. It is more compatible with large data set with multiple feature.

Like XGBoost, lightGBM operates upon maximizing information gain. However, xgboost grows trees depth-wise, while LightGBM grows trees leaf-wise.

Traditional gradient boosting decision tree need to scan all the data instances for every feature to estimate the best split points, making processing big data time-consuming and also it requires much memory. Instead, LightGBM uses histogram-based algorithm. It puts continuous feature values into discrete bins, thus increasing the training speed and model efficiency.

To further increase the speed of model training, Gradient-based One-Side Sampling(GOSS) method was introduced. LightGBM reduces the complexity of training procedure by keeping the data points with greater gradients and performs random sampling on data points with smaller gradients. To conpensate that part of data loss, GOSS introduces a constant multiplier for the data instances with small gradients.

This algorithm introduced another method called exclusive feature bundling to elaborate its compatibility with big data sets by reducing the number features. This method is based on the observation that for big data, the features are usually very sparse that they are mutually exclusive. Thus, we can bundle these mutually exclusive features together. The bundle will be treated as a single data point without any loss of information.

Now, let's test the lightGBM algorithm provided by Microsoft on the concrete dataset.

First, import the lightGBM library

```
import lightgbm as lgb
```
Here I use cross-validated mean squared error to check the accuracy of the model.
```
def LightGBM(X,y):
    mse_lightGBM = []
    for i in range(5):
      kf = KFold(n_splits=10,shuffle=True,random_state=i)
      for idxtrain, idxtest in kf.split(X):
          xtrain = X[idxtrain]
          ytrain = y[idxtrain]
          ytest = y[idxtest]
          xtest = X[idxtest]
          xtrain = scale.fit_transform(xtrain)
          xtest = scale.transform(xtest)

          gbm.fit(xtrain,ytrain)
          yhat = gbm.predict(xtest)

          mse_lightGBM.append(mse(ytest,yhat))
    return np.mean(mse_lightGBM)
```
Here we initialize the parameter values. I modified the values of number of leaves and learning rate to find a relatively good parameters for the model.
```
num_leaves_range = np.linspace(50,150,11)
learning_rate_range = np.linspace(0.001,0.1,10)
all_mse = []
for i in num_leaves_range:
  for j in learning_rate_range:
     hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves':int(i),
    'learning_rate':j}
     gbm=lgb.LGBMRegressor(**hyper_params)
     all_mse.append(LightGBM(X,y))
```
The following part is devoted to find the smallest MSE and the parameters that will yield that value
```
M =np.array(all_mse).reshape(10,11) 
print(np.min(M))
print(np.where(M==np.min(M)))
```
For the smallest MSE, we get 157.01283130985115. This is smaller than the previous multiple boosting algorithm, indicating this algorithm will provide a higher accuracy. 
Also, notice that light GBM took much less time to run.



References:
Guolin Ke; Qi Meng; Thomas Finely; Taifeng Wang; Wei Chen; Weidong Ma; Qiwei Ye; Tie-Yan Liu (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm#Light-GBM

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/



