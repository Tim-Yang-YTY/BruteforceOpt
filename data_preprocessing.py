import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
def main():
    # Select and scale numerical variables
    df = pd.read_csv('test.csv')
    df = df.drop('VEH_COULEUR_CAT', axis=1)
    df = df.drop('VEH_GARE_CODE_POSTAL_CAT',axis=1)
    df = df.where(df.notna(), None)

    unqiue_alcohol = list(pd.unique(df[['RECL_IMPLICATION_ALCOOL_DROGUE_CAT']].values.ravel()))
    alcohol = list(df[['RECL_IMPLICATION_ALCOOL_DROGUE_CAT']].values.ravel())
    alcohol_digit = []
    for brand in alcohol:
        location = unqiue_alcohol.index(brand)
        alcohol_digit.append(location)
    df["RECL_IMPLICATION_ALCOOL_DROGUE_CAT"] = alcohol_digit


    unqiue_brands = list(pd.unique(df[['VEH_MANUFACTURIER_CAT']].values.ravel()))
    car_brands = list(df[['VEH_MANUFACTURIER_CAT']].values.ravel())
    car_brands_digit = []
    for brand in car_brands:
        location = unqiue_brands.index(brand)
        car_brands_digit.append(location)
    df["VEH_MANUFACTURIER_CAT"] = car_brands_digit

    unqiue_transmission = list(pd.unique(df[['VEH_TRANSMISSION_CAT']].values.ravel()))
    car_transmission = list(df[['VEH_TRANSMISSION_CAT']].values.ravel())
    car_transmimssion_digit  = []
    for trans in car_transmission:
        location = unqiue_transmission.index(trans)
        car_transmimssion_digit.append(location)
    df["VEH_TRANSMISSION_CAT"] = car_transmimssion_digit


    unqiue_body = list(pd.unique(df[['VEH_STYLE_CARR_CAT']].values.ravel()))
    unqiue_body = [x for x in unqiue_body if x is not None]
    car_body = list(df[['VEH_STYLE_CARR_CAT']].values.ravel())
    car_body_digit = []
    for body in car_body:
        if body == None:
            car_body_digit.append(None)
        else:
            location = unqiue_body.index(body)
            car_body_digit.append(location)
    df["VEH_STYLE_CARR_CAT"] = car_body_digit

    unique_convertible = list(pd.unique(df[['VEH_DECAPOTABLE_IND']].values.ravel()))
    unique_convertible = [x for x in unique_convertible if x is not None]
    convertible = list(df[['VEH_DECAPOTABLE_IND']].values.ravel())
    convertible_digit = []
    for body in convertible:
        if body == None:
            convertible_digit.append(None)
        else:
            location = unique_convertible.index(body)
            convertible_digit.append(location)
    df["VEH_DECAPOTABLE_IND"] = convertible_digit

    unique_camera = list(pd.unique(df[['VEH_CAMERA_IND']].values.ravel()))
    unique_camera = [x for x in unique_camera if x is not None]
    camera = list(df[['VEH_CAMERA_IND']].values.ravel())
    camera_digit = []
    for body in camera:
        if body == None:
            camera_digit.append(None)
        else:
            location = unique_camera.index(body)
            camera_digit.append(location)
    df["VEH_CAMERA_IND"] = camera_digit

    unique_parking = list(pd.unique(df[['VEH_STATIONNEMENT_AUTO_IND']].values.ravel()))
    unique_parking = [x for x in unique_parking if x is not None]
    parking = list(df[['VEH_STATIONNEMENT_AUTO_IND']].values.ravel())
    parking_digit = []
    for body in parking:
        if body == None:
            parking_digit.append(None)
        else:
            location = unique_parking.index(body)
            parking_digit.append(location)
    df["VEH_STATIONNEMENT_AUTO_IND"] = parking_digit


    unique_municiple = list(pd.unique(df[['VEH_GARE_MUNICIPALITE_CAT']].values.ravel()))
    unique_municiple = [x for x in unique_municiple if x is not None]
    municiple = list(df[['VEH_GARE_MUNICIPALITE_CAT']].values.ravel())
    municiple_digit = []
    for body in municiple:
        if body == None:
            municiple_digit.append(None)
        else:
            location = unique_municiple.index(body)
            municiple_digit.append(location)

    df["VEH_GARE_MUNICIPALITE_CAT"] = municiple_digit

    unique_province = list(pd.unique(df[['VEH_GARE_PROVINCE_CAT']].values.ravel()))
    unique_province = [x for x in unique_province if x is not None]
    province = list(df[['VEH_GARE_PROVINCE_CAT']].values.ravel())
    province_digit = []
    for body in province:
        if body == None:
            province_digit.append(None)
        else:
            location = unique_province.index(body)
            province_digit.append(location)

    df["VEH_GARE_PROVINCE_CAT"] = province_digit

    unique_group = list(pd.unique(df[['COND_GROUPE_CAT']].values.ravel()))
    unique_group = [x for x in unique_group if x is not None]
    group = list(df[['COND_GROUPE_CAT']].values.ravel())
    group_digit = []
    for body in group:
        if body == None:
            group_digit.append(None)
        else:
            location = unique_group.index(body)
            group_digit.append(location)
    df["COND_GROUPE_CAT"] = group_digit

    unique_claim = list(pd.unique(df[['COND_RECL_ANNEE_PREC_IND']].values.ravel()))
    unique_claim = [x for x in unique_claim if x is not None]
    claim = list(df[['COND_RECL_ANNEE_PREC_IND']].values.ravel())
    claim_digit = []
    for body in claim:
        if body == None:
            claim_digit.append(None)
        else:
            location = unique_claim.index(body)
            claim_digit.append(location)
    df["COND_RECL_ANNEE_PREC_IND"] = claim_digit

    unique_losstype = list(pd.unique(df[['RECL_TYPE_SINISTRE_CAT']].values.ravel()))
    unique_losstype = [x for x in unique_losstype if x is not None]
    losstype = list(df[['RECL_TYPE_SINISTRE_CAT']].values.ravel())
    losstype_digit = []
    for body in losstype:
        if body == None:
            losstype_digit.append(None)
        else:
            location = unique_losstype.index(body)
            losstype_digit.append(location)

    df["RECL_TYPE_SINISTRE_CAT"] = losstype_digit




    # Convert date to int in the format YYYYMMDD


    df.to_csv('test_cleaned.csv')



# nan



    # numerical_data = data[['Distance in kilometers', 'Vehicle age', ...]]
    # scaled_data = StandardScaler().fit_transform(numerical_data)
    #
    # pca = PCA(n_components=5)  # or any other desired number of components
    # reduced_numerical_data = pca.fit_transform(scaled_data)

if __name__ == "__main__":
    main()