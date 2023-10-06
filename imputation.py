import pandas as pd
from sklearn.impute import KNNImputer
import datetime



def datetime2int(x):
    res = []
    for val in x:
        if val:
            res.append(int(str(val).replace('-', '').replace(':', '').replace(' ', '')))
        else:
            res.append(None)
    return res

def main():
    df = pd.read_csv('test_cleaned.csv')
    df = df.drop("RECL_ID_CAT",axis=1)
    df = df.drop("RECL_DATE_RAPPORTEE_DT", axis=1)

    df = df.where(df.notna(), None)
    # df['RECL_DATE_SINISTRE_DT'] = df['RECL_DATE_SINISTRE_DT'].astype(str).str.replace('-', '').str.replace(':', '').str.replace(' ', '').astype(int)
    # df['COND_PERMIS_DATE_DT'] = df['COND_PERMIS_DATE_DT'].astype(str).str.replace('-', '').str.replace(':','').str.replace(' ', '').astype(int)
    df['RECL_DATE_SINISTRE_DT'] = datetime2int(df['RECL_DATE_SINISTRE_DT'].tolist())

    df['COND_PERMIS_DATE_DT'] = datetime2int(df['COND_PERMIS_DATE_DT'].tolist())

    unqiue_year = list(pd.unique(df[['POL_NB_COND_TOTAL_CAT']].values.ravel()))

    year = list(df[['POL_NB_COND_TOTAL_CAT']].values.ravel())
    year_digit = []
    for brand in year:
        if brand != '3+':
            year_digit.append(int(brand))
        else:
            year_digit.append(3)
    df["POL_NB_COND_TOTAL_CAT"] = year_digit




    # KNN imputation
    knn_imputer = KNNImputer(n_neighbors=3)  # Use 2 nearest neighbors
    df_imputed = knn_imputer.fit_transform(df)

    # Convert back to DataFrame (optional)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    df_imputed.to_csv('test_imputed.csv')
    print(df_imputed.head())


if __name__ == "__main__":
    main()