
### DateTime format

Different time format:
```python
historic_data.aibt.dropna().apply(len).value_counts()
```
15    65665
16    46439
14    30096
13     4149
4       159
Name: aibt, dtype: int64

4 characters values corresponds to an error in the database

```python
historic_data.dropna()[historic_data.aibt.dropna().apply(len)==4]
```

### Aircrafts 

Some aircraft models were listed multiple times with different manufacturers. We decided to filter out data with *tbd* value in *# Engines* column. 
 
### Time Series invalidation



```python
df.flight.value_counts().mean() 
# 38

df.flight.value_counts().median() 
# 22
```



Trop faible pour faire des time series


### NAs acTypes 

We have multiple XXX acTypes. Two strategies are available, either we impute these values by the most frequent one for a specific flight and a specific carrier or we take the first non-null acType closest in date with the same carrier and flight.


### Particular dates treatment

Exclude special days such as Christmas, 1st of May, etc. 


### Columns to ignore

- plb_off
- eobt
- aobt
- atot

- chocks_on: future variable, linked to aibt, not used for training
- Date_Completed
- Model (To check)
- Physical Class (Engine)
- Wake Category
- Years Manufactured
- Note

### Features engineering 

carrier: One-Hot Encoding
flight: Mean-Target encoding
sto: ?
runway: One-Hot Encoding
stand: Mean-Ta****rget encoding
aldt: Quantile + Hours + Planes on same hour same weekday + weekday dummy (7 columns)
error_in_seconds: To be used for preprocessing mean target of carrier, acType, stand, runway

Manufacturer: One-Hot Encoding
\# Engines: Label Encoding
AAC: One-Hot Encoding (Relevance to check)
ADG: One-Hot Encoding (Relevance to check)
TDG: One-Hot Encoding (Relevance to check)
Wingtip Configuration: One-Hot Encoding
Main Gear Config: One-Hot Encoding
ICAO Code: One-Hot Encoding
ATCT Weight Class: One-Hot Encoding








