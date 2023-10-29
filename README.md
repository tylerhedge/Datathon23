# Datathon23

# data cleaning process:
1. Keep the continuous data with the 10 highest explained variances. Drop all other continuous data (Variances obtained by PCA testing)
2. One-hot encode relevant categorical data
3. Handle NaNs
  a. For continuous data, impute the value (replace with the mean of the column)
  b. For categorical data replace with 0
4. Reject outliers more than 1.5 times the interquartile range outside of the 1st and 3rd quartiles respectively

# presentation

https://docs.google.com/presentation/d/1EdevS4R7D2AxkVLtlzlfMmMZ6IpnEygL3w-FGjIXXhU/edit#slide=id.g4dfce81f19_0_45

