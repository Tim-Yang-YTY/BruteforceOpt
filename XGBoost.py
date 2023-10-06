import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
def main():


    data = pd.read_csv('test_imputed.csv')
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('Unnamed: 0.1', axis=1)
    scaler = StandardScaler()

    X = data
    # # X = scaler.fit_transform(X)
    # y = data['RECL_COUT_REPARATION_NUM']
    # # y = scaler.fit_transform(Y)
    #
    # # Split data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    #
    # # Convert data to DMatrix format
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)

    # param_grid = {'subsample': 1.0, 'objective': 'reg:absoluteerror', 'n_estimators': 500, 'min_child_weight': 2, 'max_depth': 7, 'learning_rate': 0.05, 'colsample_bytree': 0.9}
    # param = {
    #     'max_depth': 7,
    #     'eta': 0.3,
    #     'objective': 'reg:squarederror'
    # }
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1, 0.3],
    #     'max_depth': [3, 4, 5, 6, 7],
    #     'min_child_weight': [1, 2, 3, 4],
    #     'subsample': [0.5,0.6, 0.7, 0.9, 1.0],
    #     'colsample_bytree': [0.4, 0.5, 0.7, 0.9, 1.0],
    #     'n_estimators': [100, 200, 500,1000],
    #     'gamma': [0, 0.03, 0.1, 0.3],
    #     'reg_alpha': [1e-5, 1e-2, 0.75],
    #     'reg_lambda': [1e-5, 1e-2, 0.45],
    #     'objective': ["reg:squarederror"
    #  ,"reg:squaredlogerror"
    #  ]
    # }
    #
    #
    # # Initialize XGBoost regressor
    # model = xgb.XGBRegressor()
    #
    # # Random search
    # random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100,
    #                                    scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=3)
    # random_search.fit(X_train, y_train)
    #
    # # Print best parameters
    # print(random_search.best_params_)
    #
    # exit()


    # Train model
    num_round = 10000
    param = {'subsample': 0.7, 'reg_lambda': 1e-05, 'reg_alpha': 1e-05, 'objective': 'reg:squarederror', 'n_estimators': 100, 'min_child_weight': 4, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 0.4}
    # bst = xgb.train(param, dtrain, num_round)
    # bst.save_model('xgb_model.json')



    # Predict
    bst = xgb.XGBRegressor()
    bst.load_model('xgb_model.json')
    y_pred = bst.predict(X)

    pd.DataFrame({"PREDICT": y_pred}).to_csv("test_pred.csv")

    # print(type(y_test))



    # y_pred = loaded_model.predict(X_test)

    # Evaluate
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()
