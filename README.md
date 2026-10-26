# Titanic 



### Przygotowanie zbioru danych



```python
import arff
import numpy as np
import pandas as pd

with open('Zbiór danych Titanic.arff', 'r') as f:
    data = arff.load(f)

df = pd.DataFrame(
    data['data'],
    columns=[attr[0] for attr in data['attributes']]
)

wiek_dziesietny = pd.concat([df.loc[df['age'].notna() & (df['age'] % 1 != 0), 'age']], axis=1)
wiek_dziesietny.columns = ["Wiek"]

display(df.head(20))
display(df.describe())
display(wiek_dziesietny.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>2</td>
      <td>NaN</td>
      <td>St Louis, MO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>11</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Mr. Hudson Joshua Creighton</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>135.0</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1</td>
      <td>Anderson, Mr. Harry</td>
      <td>male</td>
      <td>48.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19952</td>
      <td>26.5500</td>
      <td>E12</td>
      <td>S</td>
      <td>3</td>
      <td>NaN</td>
      <td>New York, NY</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>1</td>
      <td>Andrews, Miss. Kornelia Theodosia</td>
      <td>female</td>
      <td>63.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13502</td>
      <td>77.9583</td>
      <td>D7</td>
      <td>S</td>
      <td>10</td>
      <td>NaN</td>
      <td>Hudson, NY</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>112050</td>
      <td>0.0000</td>
      <td>A36</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Belfast, NI</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>1</td>
      <td>Appleton, Mrs. Edward Dale (Charlotte Lamson)</td>
      <td>female</td>
      <td>53.0000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>11769</td>
      <td>51.4792</td>
      <td>C101</td>
      <td>S</td>
      <td>D</td>
      <td>NaN</td>
      <td>Bayside, Queens, NY</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>Montevideo, Uruguay</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0</td>
      <td>Astor, Col. John Jacob</td>
      <td>male</td>
      <td>47.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C62 C64</td>
      <td>C</td>
      <td>NaN</td>
      <td>124.0</td>
      <td>New York, NY</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>1</td>
      <td>Astor, Mrs. John Jacob (Madeleine Talmadge Force)</td>
      <td>female</td>
      <td>18.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>C62 C64</td>
      <td>C</td>
      <td>4</td>
      <td>NaN</td>
      <td>New York, NY</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>1</td>
      <td>Aubart, Mme. Leontine Pauline</td>
      <td>female</td>
      <td>24.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PC 17477</td>
      <td>69.3000</td>
      <td>B35</td>
      <td>C</td>
      <td>9</td>
      <td>NaN</td>
      <td>Paris, France</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>1</td>
      <td>Barber, Miss. Ellen 'Nellie'</td>
      <td>female</td>
      <td>26.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19877</td>
      <td>78.8500</td>
      <td>NaN</td>
      <td>S</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
      <td>B</td>
      <td>NaN</td>
      <td>Hessle, Yorks</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>0</td>
      <td>Baumann, Mr. John D</td>
      <td>male</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PC 17318</td>
      <td>25.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New York, NY</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.0</td>
      <td>0</td>
      <td>Baxter, Mr. Quigg Edmond</td>
      <td>male</td>
      <td>24.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>1</td>
      <td>Baxter, Mrs. James (Helene DeLaudeniere Chaput)</td>
      <td>female</td>
      <td>50.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>6</td>
      <td>NaN</td>
      <td>Montreal, PQ</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.0</td>
      <td>1</td>
      <td>Bazzani, Miss. Albina</td>
      <td>female</td>
      <td>32.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11813</td>
      <td>76.2917</td>
      <td>D15</td>
      <td>C</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>0</td>
      <td>Beattie, Mr. Thomson</td>
      <td>male</td>
      <td>36.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13050</td>
      <td>75.2417</td>
      <td>C6</td>
      <td>C</td>
      <td>A</td>
      <td>NaN</td>
      <td>Winnipeg, MN</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>1046.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1308.000000</td>
      <td>121.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.294882</td>
      <td>29.881135</td>
      <td>0.498854</td>
      <td>0.385027</td>
      <td>33.295479</td>
      <td>160.809917</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.837836</td>
      <td>14.413500</td>
      <td>1.041658</td>
      <td>0.865560</td>
      <td>51.758668</td>
      <td>97.696922</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.166700</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>155.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.275000</td>
      <td>256.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
      <td>328.000000</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wiek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.9167</td>
    </tr>
    <tr>
      <th>173</th>
      <td>32.5000</td>
    </tr>
    <tr>
      <th>222</th>
      <td>28.5000</td>
    </tr>
    <tr>
      <th>224</th>
      <td>45.5000</td>
    </tr>
    <tr>
      <th>359</th>
      <td>0.8333</td>
    </tr>
    <tr>
      <th>427</th>
      <td>0.6667</td>
    </tr>
    <tr>
      <th>512</th>
      <td>32.5000</td>
    </tr>
    <tr>
      <th>516</th>
      <td>36.5000</td>
    </tr>
    <tr>
      <th>548</th>
      <td>0.8333</td>
    </tr>
    <tr>
      <th>568</th>
      <td>18.5000</td>
    </tr>
  </tbody>
</table>
</div>


W zbiorze Titanic jest 14 cech. Obejmują one informacje o:
- pclass - klasie biletu pasażera (1/2/3 klasa)
- survived - czy przeżył (binarne 1/0, 1-przeżył, 0-nie przeżył)
- name - informacje o pasażerze w formacie (nazwisko rodzinne, Tytuł. imię)
- sex - płeć pasażera (female/male)
- age - wiek pasażera (jak można zobaczyć poniżej pierwszej tabeli przypadki gdzie wiek nie jest liczbą całkowitą. Jeden gdzie wiek jest mniejszy od 1 - prawdopodobnie dzieci, które nie ukończyły pierwszego roku życia oraz przypadki, gdzie wiek jest w formacie __.5 - prawdopodobnie są to wartości estymowane)
- sibsp - ilość rodzeństwa pasażera na statku 
- parch - liczba rodziców lub dzieci pasażera na statku
- ticket - numer biletu pasażera
- fare - cena biletu pasażera (z drugiej tabeli możemy zauważyć, że najtańszy bilet/y był za 0 dolarów - prawdopodobnie będą to osoby powiązane ze statkiem np. właściciel albo osoby specjalne, które dostały zaproszenie)
- cabin - numer kabiny pasażera
- embarked - miejsce, w którym pasażer wsiadł na statek
- boat - identyfikator łodzi, którą uratowano pasażera
- body - identyfikator zwłok przypisany pasażerowi, który nie przeżył
- home.dest - informacje o miejscu zamieszkania i miejscu docelowym pasażera

### Brakujące wartości

W pliku ARFF braki danych są oznaczane symbolem '?'. Podczas wczytywania danych za pomocą biblioteki *arff* zostały one automatycznie zinterpretowane jako NaN. W związku z tym nie ma konieczności modyfikowania danych przed użyciem funkcji sprawdzających braki w zbiorze danych.


```python
print(f"Ilość '?' w zbiorze danych: {(df == "?").sum().sum()}")
brak_ilo = df.isnull().sum()
brak_pro = df.isnull().mean() * 100 
display(pd.concat(
    [brak_ilo.rename('Ilość brakujących danych'), 
     brak_pro.rename('Procent brakujących danych [%]')], axis=1))
```

    Ilość '?' w zbiorze danych: 0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ilość brakujących danych</th>
      <th>Procent brakujących danych [%]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pclass</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>survived</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>name</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>age</th>
      <td>263</td>
      <td>20.091673</td>
    </tr>
    <tr>
      <th>sibsp</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>parch</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ticket</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>fare</th>
      <td>1</td>
      <td>0.076394</td>
    </tr>
    <tr>
      <th>cabin</th>
      <td>1014</td>
      <td>77.463713</td>
    </tr>
    <tr>
      <th>embarked</th>
      <td>2</td>
      <td>0.152788</td>
    </tr>
    <tr>
      <th>boat</th>
      <td>823</td>
      <td>62.872422</td>
    </tr>
    <tr>
      <th>body</th>
      <td>1188</td>
      <td>90.756303</td>
    </tr>
    <tr>
      <th>home.dest</th>
      <td>564</td>
      <td>43.086325</td>
    </tr>
  </tbody>
</table>
</div>



```python
dfnull = df.copy()

for col in brak_ilo[brak_ilo > 0].index:
    if(f'{col}Null' not in df.columns):
        df[f"{col}Null"] = np.where(df[col].isnull(), 1, 0)
        #Jedna linijka
        # display(df.groupby('survived')[col].apply(lambda x: x.isnull().mean()))

display(df)

df_map = pd.DataFrame()
for col in df.columns[len(dfnull.columns):]:
    df_map = pd.concat([df_map, df.groupby('survived')[col].mean() * 100], axis=1)
display(df_map)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>...</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
      <th>ageNull</th>
      <th>fareNull</th>
      <th>cabinNull</th>
      <th>embarkedNull</th>
      <th>boatNull</th>
      <th>bodyNull</th>
      <th>home.destNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>...</td>
      <td>2</td>
      <td>NaN</td>
      <td>St Louis, MO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>...</td>
      <td>11</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Mr. Hudson Joshua Creighton</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>...</td>
      <td>NaN</td>
      <td>135.0</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>3.0</td>
      <td>0</td>
      <td>Zabour, Miss. Hileni</td>
      <td>female</td>
      <td>14.5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2665</td>
      <td>14.4542</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>328.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>3.0</td>
      <td>0</td>
      <td>Zabour, Miss. Thamine</td>
      <td>female</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2665</td>
      <td>14.4542</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>3.0</td>
      <td>0</td>
      <td>Zakarian, Mr. Mapriededer</td>
      <td>male</td>
      <td>26.5000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2656</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>3.0</td>
      <td>0</td>
      <td>Zakarian, Mr. Ortin</td>
      <td>male</td>
      <td>27.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2670</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>3.0</td>
      <td>0</td>
      <td>Zimmerman, Mr. Leo</td>
      <td>male</td>
      <td>29.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>315082</td>
      <td>7.8750</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1309 rows × 21 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ageNull</th>
      <th>fareNull</th>
      <th>cabinNull</th>
      <th>embarkedNull</th>
      <th>boatNull</th>
      <th>bodyNull</th>
      <th>home.destNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.485785</td>
      <td>0.123609</td>
      <td>87.391842</td>
      <td>0.0</td>
      <td>98.887515</td>
      <td>85.043263</td>
      <td>50.803461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.600000</td>
      <td>0.000000</td>
      <td>61.400000</td>
      <td>0.4</td>
      <td>4.600000</td>
      <td>100.000000</td>
      <td>30.600000</td>
    </tr>
  </tbody>
</table>
</div>


### Rodzaje brakujących danych

- ageNull - *MAR* - prawdopodobieństwo nieznania wieku, mogło wynikać z klasy biletu.
- fareNull - *MCAR* - bardzo mała wartość braku danych, raczej jest to pojedyncze przeoczenie.
- cabinNull - *MAR* - brak numeru kabiny jest skorelowany z klasą biletu, ceną biletu i przeżywalnością - osoby z wyższą klasą biletu oraz społeczną miały większe szanse na ratunek.
- embarkedNull - *MCAR* - bardzo mała wartość braku danych, są to pojedyncze przeoczenia.
- boatNull - *MNAR* - wartość w tej kolumnie zależy od tego czy osoba skorzystała z łodzi ratunkowej, która de-facto w dużym stopniu mówi o tym czy ta osoba przeżyła.
- bodyNull - *MNAR* - wartość w tej kolumnie zależy od tego czy osoba przeżyła.
- home.destNull - *MAR* - informacje o miejscu docelowym zbierano od osób wyższych klas i tych które przeżyły.

### Co zrobić z brakującymi danymi

- age - uzupełnienie związane z innymi cechami np. name - w tej cesze mamy informację o tytule osoby, więc możemy je uzupełnić np. medianą danego tytułu
- fare - jeden przypadek, gdzie nie ma tej informacji; można usunąć wiersz - nie utrudni to dalszej analizy 
- cabin - dużo braków; można dla wierszy gdzie mamy informacje o kabinie zostawić pierwszą literę dla łatwiejszej kategoryzacji, a w przypadku braku zastąpić to np. literą "U" (Unknown)
- embarked - dwa przypadki, gdzie nie ma tej informacji; można usunąć wiersze - nie utrudni to dalszej analizy 
- boat - usunięcie całej kolumny; informacja bezpośrednio związana z przeżyciem - nie nadaje się do modelowania, zaburzyłaby cel modelu
- body - usunięcie całej kolumny; informacja bezpośrednio związana z przeżyciem - nie nadaje się do modelowania, zaburzyłaby cel modelu
- home.dest - dużo braków; można dla wierszy gdzie mamy informacje o destynacji zostawić tylko państwo, a w przypadku braku zastąpić to np. literą "U" (Unknown)
