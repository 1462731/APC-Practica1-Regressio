import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Carreguem el dataset sencer
rahan_csv = pd.read_csv('dataset/Rahan.csv', delimiter=';')
rahan_csv = rahan_csv.iloc[:,:-2]
abrasan_csv = pd.read_csv('dataset/TabrizPollution/Abrasan.csv', delimiter=';')
bashumal_csv = pd.read_csv('dataset/TabrizPollution/Bashumal.csv', delimiter=';')
rastakucha_csv = pd.read_csv('dataset/TabrizPollution/RastaKucha.csv', delimiter=';')
data = pd.concat([abrasan_csv, bashumal_csv, rastakucha_csv], ignore_index=True)
# Veiem una mica les dades
print(data.head())

data.info()
data.describe()

# Netejem tots els valors nuls
df = data[(data['air_temperature'] != -9999.0) & (data['dewpoint'] != -9999.0) & (data['wind_direction_corr'] != -9999.0) &
            (data['wind_speed'] != -9999.0) & (data['relative_pressure'] != -9999.0) & (data['PM10'] != -9999.0) &
            (data['PM2.5'] != -9999.0)]
df.describe()

# Les mostrem en gràfics pairplot, per veure com es relacionen
sns.pairplot(df)
plt.figure()

# Mostrem la matriu de correlació amb el mateix objectiu
correlation = df.corr()
plt.figure()

ax = sns.heatmap(correlation, annot=True, linewidths=.5)

# Normalitzem les dades i fem el mateix
normalized_df=(df-df.min())/(df.max()-df.min())

correlation = normalized_df.corr()
plt.figure()

ax = sns.heatmap(correlation, annot=True, linewidths=.5)
sns.pairplot(normalized_df)
plt.figure()

# Com que les relacions entre variables dos a dos no eren molt clares, veiem relacions amb varaibles tres a tres.
sns.pairplot(data=df.sample(1000), hue='PM2.5')
plt.figure()

sns.pairplot(data=df.sample(15000), hue='PM10')
plt.figure()






