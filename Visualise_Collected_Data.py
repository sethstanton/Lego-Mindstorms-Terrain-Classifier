import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip() and '[' in line and ']' in line:
                line_data = line.strip().strip('[]').split(', ')
                data.append([float(val) for val in line_data])
    df = pd.DataFrame(data, columns=['MotorA', 'MotorB', 'MotorC', 'MotorD', 'Pitch', 'Roll', 'AccX', 'AccY', 'AccZ',
                                     'AngVelPitch', 'AngVelRoll', 'AngVelYaw'])#, 'DownSensor', 'FrontSensor'
                                         # ^^ These should be added into here using data from second round data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df



def plot_data(data, title):
    sensor_types = {
        'Motor Encoder Positions': ['MotorA', 'MotorB', 'MotorC', 'MotorD'],
        'Tilt Angles': ['Pitch', 'Roll'],
        'Acceleration Readings': ['AccX', 'AccY', 'AccZ'],
        'Angular Velocity Readings': ['AngVelPitch', 'AngVelRoll', 'AngVelYaw'],
         #'Ultrasonic Sensor Readings' : ['DownSensor', 'FrontSensor'] -
         # ^^ This line needs to be uncommented when using data from second round data
    }

    for sensor_type, columns in sensor_types.items():
        plt.figure()
        for column in columns:
            plt.plot(data[column], label=column)
        plt.title(f'{title} - {sensor_type}', fontsize=24)
        plt.legend()
        plt.xlabel('Sample',fontsize=18)
        plt.ylabel('Reading', fontsize=18)
    plt.show()
# This makes use of the first round data, change the name to load in other data
data = load_data('FirstRound/fakewood2.txt')


# This makes use of the second round data, change the name to load in other data
# data = load_data('SecondRound/Forward/Wood2.txt')
plot_data(data,'Wood Data')
