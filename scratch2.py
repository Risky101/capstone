import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 as psycopg2
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

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_dataset(data)

X = data[['bene_count', 'total_claim_count', 'total_30_day_fill_count', 'total_day_supply', 'total_drug_cost', 'bene_count_ge65', 'total_claim_count_ge65', 'total_30_day_fill_count_ge65', 'total_day_supply_ge65', 'total_drug_cost_ge65']]
y = data['fraud']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.38,random_state=0)

rfa = RandomForestClassifier(n_estimators=100)
rfa.fit(X_train,y_train)
y_pred = rfa.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print (X_test) #test dataset (without the actual outcome)
print (y_pred) #predicted values
print()
print()

plt.show()

featureImportances = pd.Series(rfa.feature_importances_).sort_values(ascending=False)
print(featureImportances)

sns.barplot(x=round(featureImportances,5), y=featureImportances)
plt.xlabel('Features Importance')
plt.show()

print()
print()

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

try:
    con = psycopg2.connect("dbname=testdb user=postgres password=11111111")
    cur = con.cursor()
    query = " SELECT npi,bene_count,total_claim_count,total_30_day_fill_count,total_day_supply,total_drug_cost,bene_count_ge65,total_claim_count_ge65,total_30_day_fill_count_ge65,total_day_supply_ge65,total_drug_cost_ge65 FROM public.part_d ORDER BY Random() LIMIT 1; "
    cur.execute(query)
    row = cur.fetchone()
    
    print(row)
    print()
    holder = row[0]
    checker = row[1:]
    print()
    print(holder)
    print()
    print(checker)
    
    checker = pd.DataFrame(checker)
    missingdata(checker)
    clean_dataset(checker)
    
    
    
    prediction = rfa.predict(checker)
    print()
    print('Prediction result: ', prediction)
    
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if con is not None:
        con.close()