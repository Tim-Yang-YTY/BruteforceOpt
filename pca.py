import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # Standardize data
    df_imputed  = pd.read_csv('train_imputed.csv')
    # df_imputed = df_imputed.drop("RECL_COUT_REPARATION_NUM", axis=1)
    df_imputed = df_imputed.drop('Unnamed: 0',axis=1)
    df_imputed = df_imputed.drop('Unnamed: 0.1', axis=1)
    # print(len(df_imputed.columns))
    # exit()
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_imputed)

    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(df_standardized)
    res = []
    for i, component in enumerate(pca.components_):
        top_features_idx = np.argsort(component)[-1:]  # Top 3 features for demonstration
        top_features = [df_imputed.columns[idx] for idx in top_features_idx]
        # print(f"Principal Component {i + 1}: Top contributing features: {top_features}")
        res.append(top_features[0])
    comps = list(set((res)))
    comps.append("VEH_GARE_LATITUDE_NUM")
    # print(comps)

    df_pca = df_imputed[comps]
    df_pca.to_csv('trained_pca.csv')





    # print(principal_components)
    # # Determine the number of components for a desired cumulative variance
    # explained_variance = pca.explained_variance_ratio_
    # cumulative_variance = np.cumsum(explained_variance)
    # optimal_components = np.argmax(cumulative_variance >= 0.95) + 1  # 95% variance
    # print(f"Optimal number of components: {optimal_components}")

if __name__ == "__main__":
    main()