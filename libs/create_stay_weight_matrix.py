import numpy as np
import sys
sys.path.append('../../libs/')  # FeatureQuantization クラスが含まれるパスを追加
from train_utils import FeatureQuantization

def create_stay_weight_matrix(day_counts, quantization):
    """
    Creates a stay weight matrix based on the specified day counts and quantization parameters.

    Parameters:
    - day_counts: A list or array containing the counts of days. Expects [holiday_count, weekday_count].
    - quantization: An instance of FeatureQuantization containing quantization parameters.
    
    Returns:
    - A NumPy array representing the stay weight matrix.
    """
    if day_counts is not None:
        # Weights for imbalance between weekday and holiday counts
        dow_weight = [day_counts[1]/sum(day_counts), day_counts[0]/sum(day_counts)]

    # Weights to account for the length of each stay
    stay_weight = [1, 1, 1, 2, 3, 5, 10]

    stay_weight_matrix = []
    for dow_num in range(2):
        for dt_num in range(12):
            for e_num in range(7):
                vector = [int(0) for _ in range(quantization.quant_num)]
                for i in range(stay_weight[e_num]):
                    dt_num_ = (dt_num + i) % 12
                    token = int(0)
                    token += (quantization.dt_quant_num * quantization.e_quant_num) * dow_num
                    token += quantization.e_quant_num * dt_num_
                    token += e_num
                    vector[token] += dow_weight[dow_num]
                stay_weight_matrix.append(vector)

    return np.array(stay_weight_matrix)

if __name__ == "__main__":
    # Replace with actual day counts
    day_counts = [5, 2]

    # Initialize quantization
    quantization = FeatureQuantization()

    # Calculate the stay weight matrix
    stay_weight_matrix = create_stay_weight_matrix(day_counts, quantization)

    # Save the matrix
    np.save("../data/util_file/stay_weight_matrix.npy", stay_weight_matrix)
    print("Stay weight matrix saved to 'stay_weight_matrix.npy'")
