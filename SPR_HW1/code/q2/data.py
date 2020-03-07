import pandas as pd
import pickle

data = pd.read_csv('/Data/first_half_logs.csv', header=None)
count = len(data)

def load_data():

    user_data = {}

    for i in range(0, count):

        if not (data[1][i] in user_data):
            user_data[data[1][i]] = {}
            user_data[data[1][i]]['x'] = [data[2][i]]
            user_data[data[1][i]]['y'] = [data[3][i]]

        else:
            arr = user_data[data[1][i]]['x']
            arr.append(data[2][i])
            user_data[data[1][i]]['x'] = arr

            arr = user_data[data[1][i]]['y']
            arr.append(data[3][i])
            user_data[data[1][i]]['y'] = arr


    with open('data.pickle', 'wb') as handle:
        pickle.dump(user_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


with open('/q2/data.pickle', 'rb') as handle:
    user_data = pickle.load(handle)