import pandas as pd

df = pd.read_csv('test.csv')

EXPERT_FEE = 1500

def get_suggestion(row):
    predicted_repair_cost = row['PREDICT']
    car_value = row['VEH_VALEUR_ACTUELLE_NUM']
    cost_difference = abs(predicted_repair_cost - car_value)
    
    if predicted_repair_cost < 0.8 * car_value:
        return 0  # Repair the vehicle
    elif predicted_repair_cost > 1.2 * car_value:
        return 1  # Reimburse total value of vehicle
    elif cost_difference < EXPERT_FEE:
        # If potential cost difference is less than expert fee, 
        # opt for the cheaper option between repair and reimbursement
        return 0 if predicted_repair_cost < car_value else 1
    else:
        return 2  # Send to the expert

df['PRED'] = df.apply(get_suggestion, axis=1)

df.to_csv('test.csv', index=False)
