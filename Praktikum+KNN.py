
# coding: utf-8

# # Classification using KNN with Pima Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset

# In[2]:


data = pd.read_excel (r'F:\S2\Asdos\Data Mining\Selesai ETS\pima.xlsx', sheet_name='pima')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


sns.countplot(x=data['Class'])
plt.show()


# In[6]:


g = sns.PairGrid(data, hue="Class")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
plt.show()


# # Modelling

# ## KNN with Cross Validation 

# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# In[7]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[27]:


from sklearn.model_selection import KFold, cross_val_score
y = data['Class']
X = data.drop(['Class'], axis=1)
k_fold = KFold(n_splits=10)
akurasi = cross_val_score(knn, X, y, cv=k_fold, scoring='accuracy')
presisi = cross_val_score(knn, X, y, cv=k_fold, scoring='precision')
recallss = cross_val_score(knn, X, y, cv=k_fold, scoring='recall')


# In[28]:


print(akurasi)


# In[29]:


print(presisi)


# In[30]:


print(recallss)


# In[10]:


print(akurasi.mean())
print(presisi.mean())
print(recallss.mean())


# # AUC n ROC

# In[58]:


knn.fit(X,y)
fig, ax1 = plt.subplots(figsize=(12, 8))
    mean_tpr = 0.0
    mean_fpr = linspace(0, 1, 100)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=999)

    for i, (train_index, test_index) in enumerate(k_fold.split(X,y)):
        # calculate the probability of each class assuming it to be positive
        probas_ = knn.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random', lw=2)


# # The best K

# In[22]:


k_range = range(1, 24)
k_akurasi = []
k_presisi = []
k_recallss = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    akurasi = cross_val_score(knn, X, y, cv=k_fold, scoring='accuracy')
    k_akurasi.append(akurasi.mean())
    presisi = cross_val_score(knn, X, y, cv=k_fold, scoring='precision')
    k_presisi.append(presisi.mean())
    recallss = cross_val_score(knn, X, y, cv=k_fold, scoring='recall')
    k_recallss.append(recallss.mean())


# In[23]:


print(k_akurasi)


# In[24]:


print(k_presisi)


# In[25]:


print(k_recallss)


# In[40]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(k_range, k_akurasi, color='red')
plt.plot(k_range, k_presisi, color='blue')
plt.plot(k_range, k_recallss, color='green')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated Value')

