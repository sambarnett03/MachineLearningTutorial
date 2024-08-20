import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import default_rng
from itertools import permutations

import sys
sys.path.append('c:\\Users\\vhl49747\\OneDrive - Science and Technology Facilities Council\\Documents\\GitHub\\nqcc_summer_students_readout_sensitivity\\SCC_Control\\SCC_Control')
from scc_control.Analysis.PlotFunctions import *

def format_data(Data):
    '''Takes IQ data whose x and y components are given by the real and imaginary parts respectively and outputs one feature and one target array'''
    g = Data['state_desc'][0]
    e = Data['state_desc'][1]

    # Need one array with all the target data and a multidimensional array with the features
    # In this case, we know everything corresponding to g should be 0 and e should be 1
    target_g = np.zeros(g.shape)
    target_e = np.ones(e.shape)
    target_array = np.append(target_g, target_e)

    # For the feature array, want coordinates of each point so [[x1,y1], [x2,y2], ...] with gs coords coming first
    g = np.stack((g.real, g.imag), 1)
    e = np.stack((e.real, e.imag), 1)
    feature_array = np.vstack((g, e))   

    return feature_array, target_array





# def plot_accuracy(pred_g, pred_e, pred_inc):
#     ax = PlotFunction(pred_g[:, 0], pred_g[:, 1], 'Generic', 'State Descrimination', plotStyle= 'b.', labelx = 'Q', labely = 'I', pltorno='no')
#     ax = addtoPlot(pred_e[:, 0], pred_e[:, 1], ax, " ", plotStyle= 'r.', pltorno='no')
#     ax = addtoPlot(pred_inc[:, 0], pred_inc[:, 1], ax, " ", plotStyle= 'g.')
#     return


def plot_accuracy(predictions):
    colours = cm.rainbow(np.linspace(0, 1, len(predictions.keys())))
    data = [plt.scatter(value[:, 0], value[:, 1], color=colours[i], s=0.5, label=key) for i, (key, value) in enumerate(predictions.items())]
    plt.legend()
    plt.show()
    return
    




def categorise_data(data, prediction, known_results, no_clusters):
    predictions = {}
    prediction[prediction == -1] = 0
    known_results[known_results == -1] = 0

    for i in range(no_clusters):
        predictions[f'State {i}'] = data[(prediction == i) & (known_results == i)]
        predictions['incorrectly classified'] = data[((prediction == i) & (known_results != i)) | ((prediction != i) & (known_results == i))]
    
    acc = find_accuracy(known_results, prediction)
    return predictions, acc




def find_accuracy(target_array, predicted_array):
    return len(target_array[target_array == predicted_array]) / len(target_array)




def map_to_target(target_array, predicted_array, n):
    perms = np.array(list(permutations(range(n))))
    results = perms[:, predicted_array]
    accuracies = [find_accuracy(target_array, result) for result in results]
    prediction = results[np.argmax(accuracies)]
    return prediction, np.max(accuracies)


def rotate_data(data, theta):
    r_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.einsum('ij, kj -> ki', r_matrix, data)




def generate_data(no_data_points, no_clusters, balanced=True, output=False, cov=[[1,0],[0,1]], centres='def'):
    rng = default_rng(132)  
    if centres == 'def':
        centres = [[3,0], [-3,0], [0, 3], [0, -3], [5, 1], [-5, -1]]
    centres = centres[: no_clusters]
    data = []
    target = []

    for i in range(no_data_points):
        if balanced == False:
            int = np.random.choice(range(no_clusters))
            for j in range(no_clusters):
                if int == j:
                    if j % 2 == 0:
                        int2 = np.random.choice(range(no_clusters))
                        data.append(rng.multivariate_normal(centres[(j + int2) % no_clusters], cov))
                        target.append((j + int2) % no_clusters)

                    else:
                        data.append(rng.multivariate_normal(centres[j], cov))
                        target.append(j)

        else:
            data.append(rng.multivariate_normal(centres[i % no_clusters], cov))
            target.append(i % no_clusters)

    data = np.array(data)
    target = np.array(target)


    if output == True:
        print(f'{no_data_points} were generated in total\n')
        for i in range(no_clusters):
            print(f'Cluster {i} contains {len(target[target == i])} points\n')
  
    
    return data, target



def analyse_data(target_array, allocations, data, n, supervised, plotorno=True):
    ''' Wrapper function for map_to_target(), catergorise_data(), and plot_accuracy()'''

    if supervised == False:
        allocations, accuracy = map_to_target(target_array, allocations, n=n)

    predictions, accuracy = categorise_data(data, allocations, target_array, n)

    if plotorno == True:
        plot_accuracy(predictions)

    return accuracy

    

def split_data(feature_array, target_array, training_size):
    target_array, feature_array = np.array(target_array), np.array(feature_array)
    no_data_points = target_array.shape[0]
    rand_ints = np.random.choice(range(no_data_points), int(training_size * no_data_points), replace=False)
    
    train_target = np.array([target_array[i] for i in rand_ints])
    train_feature = np.array([feature_array[i] for i in rand_ints])


    new_ints = np.delete(range(no_data_points), rand_ints)
    test_target = np.array([target_array[i] for i in new_ints])
    test_feature = np.array([feature_array[i] for i in new_ints])

    return train_target, train_feature, test_target, test_feature




def read_s2p(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the comments and extract data
    data = []
    for line in lines:
        if not line.startswith('!') and not line.startswith('#'):
            data.append([float(x) for x in line.strip().split()])
    
    data = np.array(data)
    
    # Frequency in Hz
    frequency = data[:, 0] * 10 ** 9
    s21_real = data[:, 3]
    s21_imag = data[:, 4]
    return frequency, s21_real, s21_imag