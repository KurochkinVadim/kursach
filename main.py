import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

data = pd.read_csv('smoke_detection_iot.csv')
cols = list(data.columns)
features = cols[:-1]
target = cols[-1]
data = data.drop(['UTC', 'CNT'], axis=1)
features = features[2:-1]

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_emul, y_test, y_emul = train_test_split(X_test, y_test, test_size=0.2, random_state=4)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_emul = X_emul.sort_values('Pressure[hPa]')
X_emul_sc = scaler.transform(X_emul)

filename = 'final_model.sav'
final_model = pickle.load(open(filename, 'rb'))


class FireFightingModule:

    def __init__(self, data, model, features):
        self.visualizationData = data
        self.model = model
        self.features = features

    def dataCollection(self, externalSource):
        self.visualizationData.append(externalSource)
        if len(self.visualizationData) > 4:
            self.visualizationData.pop(0)

    def getData(self):
        return self.visualizationData

    def predict(self, data):
        predict = self.model.predict(data)
        if predict[0] == 0:
            return 'The condition is normal.'
        else:
            return 'Alarm! Fire!'

    def visualization(self):
        plt.rcParams.update({'font.size': 5, })
        fig, ax = plt.subplots(2, 6)
        ys = [[], [], [], [], [], [], [], [], [], [], [], []]
        for line in self.visualizationData:
            for i in range(12):
                ys[i].append((line[i]))
        for i in range(6):
            ax[0, i].plot(range(4), ys[i])
            ax[0, i].set_title(self.features[i])
        for i in range(6):
            ax[1, i].plot(range(4), ys[i + 6])
            ax[1, i].set_title(self.features[i + 6])
        plt.show()


def animate(i):
    global k
    global xs
    module.dataCollection(X_emul.to_numpy(dtype='int')[k])
    values = module.getData()
    ys = [[], [], [], [], [], [], [], [], [], [], [], []]

    x = datetime.now().strftime("%H:%M:%S")
    print(module.predict(X_emul_sc[k].reshape(1, -1)) + "Time: " + x)
    xs.append(x)
    if len(xs) > 4:
        xs.pop(0)

    for line in values:
        for i in range(12):
            ys[i].append((line[i]))

    for i in range(2):
        for j in range(6):
            ax[i, j].clear()

    for i in range(6):
        ax[0, i].plot(xs, ys[i])
        ax[0, i].set_title(features[i])
    for i in range(6):
        ax[1, i].plot(xs, ys[i + 6])
        ax[1, i].set_title(features[i + 6])

    plt.rc('font', size=5)
    k += 1


fig, ax = plt.subplots(2, 6)
xs = []
k = 0
externalSource = []
module = FireFightingModule(externalSource, final_model, features)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
