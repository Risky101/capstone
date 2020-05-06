import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv(r'F:\Kris\Documents\ULL\a 2020 Spring\INFX 490\Project\data\fraudlearn.csv')

data.head()

def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]

    plt.xticks(rotation='90')
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Accuracy', fontsize=15)

    return ms

missingdata(data)

print (data)
print ()
print ()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(data)

X = data[['bene_count', 'total_30_day_fill_count', 'total_day_supply', 'bene_count_ge65', 'total_claim_count_ge65', 'total_30_day_fill_count_ge65', 'total_day_supply_ge65', 'total_claim_count', 'total_drug_cost', 'total_drug_cost_ge65']]
y = data['fraud']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)

rfa = RandomForestClassifier(n_estimators=100)
rfa.fit(X_train,y_train)
y_pred = rfa.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('This is the test dataset without outcomes: ')
print (X_test) #test dataset (without the actual outcome)
print ()
print ()
print ('This is the predicted values: ')
print (y_pred) #predicted values
print()
print()

plt.show()

featureImportances = pd.Series(rfa.feature_importances_).sort_values(ascending=False)

print ('These are the importance of each object for prediction: ')
print(featureImportances)

sns.barplot(x=round(featureImportances,5), y=featureImportances)
plt.xlabel('Features Importance')
plt.show()

print()
print()

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
