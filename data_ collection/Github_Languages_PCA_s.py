
# coding: utf-8

# # Factor analysis of Languages

# In[18]:


import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd


sns.set(style="white")



def clean_data(data, ind, list_of_relevant_data):
    """
    Function cleanning spaces and make relevant variables lower case
    Input: 
        data: df
        ind: new index, variable name
        list_of_relevant_data: list of variables planned to sue for PCA
    Return: dataframe with relevant variables, with clean names
    """
    renamed={}
    data2=data.set_index(ind)
    data3=data2[list_of_relevant_data]
    new_names=[c.replace(' ','_').lower() for c in data3.columns]
    for (old, new) in zip(user_fields.columns, new_names):
        renames[old]=new
    data4=data3.rename(columns=renamed)
    return data4


def create_formula(list_of_relevant_data):
    """
    create a formula for OLS dependent variable part to generate analysis
    """

    formula=' ~ '
    for i in new_names:
        formula=formula+'+'+str(i)
    return formula


def show_plot(explained_variance, datapath):
    plt.figure(figsize=(6,7))
    plt.plot(explained_variance, linestyle='--',color='r', label='Cumulative Variance Explained')
    plt.bar(np.arange(len(var)),var, width=0.6, alpha=0.7,
                     color='grey',
                     label='Explained Variance by Componensts')

    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance Ratios by PCs')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks( np.arange(len(var)) + 0.01, ['PC %d' %p for p in range(1,len(var))])
    plt.tight_layout()
    sns.despine()
    plt.savefig(datapath+"Variable_importance_PCA.png")   #save the figure to file
    plt.show()
    plt.close()



    plt.figure(figsize=(6,7))
ax=a['Sample-1'].sort_values().plot(kind='barh', label='Explained Variance')
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
plt.plot(a['Sample-1'].sort_values().cumsum(), np.linspace(0,6,6), linestyle='--', label='Cumulative Variance Explained')
plt.legend(loc=4)
sns.despine(bottom=True, top=False)
ax.xaxis.tick_top()
plt.tight_layout()
plt.savefig(pathdataout+"PCA_Sample1_explained_variance.pdf")  



def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(0,q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)


nc=6

def run_pca_components(pathdata, data, ind, list_of_relevant_data, nc):
    """
    Run Principal Component Analysis
    Inputs:
        pathdata: path to dataframe
        data: df
        nc: number of components
    """
    data=pd.read_csv(pathdataout+data)
    d1=clean_data(data, ind, list_of_relevant_data)
    print(len(d1.columns))
    X=d1.values
    X=varimax(X, gamma = 1.0, q = 20, tol = 1e-6)
    pca = PCA(n_components=nc, svd_solver="full")
    pca.fit(X)
    var= pca.explained_variance_ratio_
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    show_plot(var1, pathdataout)
    expi=[(i*100) for i in pca.explained_variance_ratio_]
    print(expi)
    print(sum(expi))
    a=pca.fit_transform(X)
    for j in range(0,nc): # save PCA_s
        v="PCA_"+str(j)
        d1[v]=[i[j] for i in a]
    # analyse variable importance and stregth with OLS on each new coponent
    for p in range(0,nc):
        formula ="PCA_"+str(p)+create_formula(list_of_relevant_data)
        result = smf.ols(formula, data=d1).fit()
        df = pd.concat((result.params, result.tvalues, result.pvalues), axis=1)
        df=df.rename(columns={0: 'beta', 1: 't', 2:"p"})
        df=df.sort(['beta'], ascending=False)
        print(df)
    
    return df