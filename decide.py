import pandas as pd

def main():
    def decide_action(predicted_repair_cost, car_value, expert_cost=1500, margin=0.13):
        """
        Decide the action based on the predicted repair cost and car value.
        :param predicted_repair_cost: The predicted cost for repairing.
        :param car_value: The car's value.
        :param expert_cost: Cost of hiring an expert.
        :param margin: Margin for prediction error.
        :return: 0 if repair, 1 if reimburse, 2 if send to expert.
        """

        if predicted_repair_cost < car_value * (1 - margin):
            return int(0)  # Repair the vehicle
        elif predicted_repair_cost > car_value * (1 + margin):
            return int(1)  # Reimburse total value of vehicle
        else:
        # If the difference between predicted repair and car value is less than expert fee, choose the safer option
            if abs(predicted_repair_cost - car_value) < expert_cost:
                return int(1) if predicted_repair_cost > car_value else int(0)
        return int(2)  # Send to the expert

    def process_csv(df):

        # Read the CSV file
        # df = pd.read_csv(file_path)

        # Apply the decision function to each row
        res = []
        ls = df.tolist()
        for l in ls:
            res.append()

        df['Suggestion'] = df.apply(lambda row: decide_action(row1['PREDICT'], row2['VEH_VALEUR_ACTUELLE_NUM']), axis=1)

        return df

    row1 = pd.read_csv("test_pred.csv")
    row2 = pd.read_csv("test.csv")

    df = pd.concat([row1,row2],axis=1)
    print(df.head())

    df_result = process_csv(df)
    print(df_result)


if __name__ == "__main__":
    main()


