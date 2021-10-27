import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import copy


def return_full_data():
    abrasan_csv = pd.read_csv('dataset/TabrizPollution/Abrasan.csv', delimiter=';')
    bashumal_csv = pd.read_csv('dataset/TabrizPollution/Bashumal.csv', delimiter=';')
    rastakucha_csv = pd.read_csv('dataset/TabrizPollution/RastaKucha.csv', delimiter=';')
    data = pd.concat([abrasan_csv, bashumal_csv, rastakucha_csv], ignore_index=True)
    df = data[(data['air_temperature'] != -9999.0) & (data['dewpoint'] != -9999.0) & (data['wind_direction_corr'] != -9999.0) &
                (data['wind_speed'] != -9999.0) & (data['relative_pressure'] != -9999.0) & (data['PM10'] != -9999.0) &
                (data['PM2.5'] != -9999.0)]
    return df

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[:]
    x_t /= std[:]
    return x_t


""" Per a assegurar-nos que el model s'ajusta be a dades noves, no vistes, 
cal evaluar-lo en un conjunt de validacio (i un altre de test en situacions reals).
Com que en aquest cas no en tenim, el generarem separant les dades en 
un 80% d'entrenament i un 20% de validació.
"""
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


class Regressor(object):
    def __init__(self, theta, alpha, w0):
        self.theta = theta
        self.alpha = alpha
        self.w0 = w0

    def cost(self, x, y):
        # funcio de cost
        m = len(y)
        predictions = self.w0 + x.dot(self.theta)
        # print(predictions)
        errors = np.subtract(predictions, y)
        # print ("Errors: ", errors)
        J = (1 / (2 * m)) * np.sum(np.square(errors))

        return J

    def predict(self, x_test):
        # implementar aqui la funció de prediccio

        return self.w0 + x_test.dot(self.theta)

    def __update(self, hy, y):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real.

        pass

    def train(self, x, y, iterations: int, epsilon=0):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        cost_history = np.zeros(iterations, dtype=object)
        m = len(y)
        for it in range(iterations):
            predictions = self.w0 + x.dot(self.theta)
            gradient = (predictions - y)
            self.w0 = self.w0 - self.alpha/m *np.sum(gradient)
            for j in range(len(self.theta)):
                final_gradient = gradient * x[:, j]
                self.theta[j] = self.theta[j] - self.alpha/m * np.sum(final_gradient)
            cost_history[it] = self.cost(x, y)
        return self.theta, cost_history


def testing(x_train, y_train, x_val, y_val, alphas, theta, w0, iterations):
    cost_history = []
    meanr2s = []
    theta1 = 0
    for j in range(len(alphas)):
        alpha = alphas[j]
        theta1 = copy.deepcopy(theta)
        regr = Regressor(theta1, alpha, w0)
        # y_train = np.zeros(4)
        theta1, cost = regr.train(x_train, y_train, iterations)
        cost_history.append(cost)
        print('Final value of theta =', theta1)
        print('First 5 values from cost_history =', cost[:5])
        print('Last 5 values from cost_history =', cost[-5:])

        acc = 0
        for i in range(len(y_val)):
            y_pred = regr.predict(x_val)
            r2score = r2_score(y_val, y_pred)
            acc = acc + r2score
        meanr2s.append(acc / len(y_val))
        print(meanr2s[j])

    plt.figure(1)
    for j in range(len(alphas)):
        plt.plot(range(1, iterations + 1), cost_history[j], label='alpha =%.5f' % alphas[j])
    plt.grid()
    plt.xlabel("Nombre iteracions")
    plt.ylabel("Cost (J)")
    plt.legend(bbox_to_anchor=(1.04, 0.4), loc="upper left")
    plt.title("Reducció del cost - {} iteracions".format(iterations), y=1.08)
    plt.show()

    plt.figure(2)
    plt.bar(['0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05'], meanr2s)
    plt.grid()
    plt.ylabel('r2 score')
    plt.xlabel('alpha value')
    plt.ylim([0.0, 1.0])
    plt.title("r2 score per learning rate - {} iteracions".format(iterations), y=1.08)
    plt.show()
    return 0


# Obtenim les dades netes i les estandaritzem
full_data = return_full_data()
standarized_data = standarize(full_data)
standarized_data.hist(bins=25)
plt.show()
standarized_data = standarized_data.drop('wind_direction_corr',axis=1)
standarized_data= standarized_data.drop('Time',axis=1)
standarized_data = standarized_data[['air_temperature', 'dewpoint', 'relative_pressure', 'wind_speed', 'PM10', 'PM2.5']]

x = standarized_data.values[:, :5]
y = standarized_data.values[:, 5]

x_train, y_train, x_val, y_val = split_data(standarized_data.values[:, :5], standarized_data.values[:, 5])

thetass = np.random.uniform(0.0, 1.0, size=5)
w0 = np.random.uniform(0.0, 1.0)
alphas = [0.0001, 0.0005, 0.001,0.005,0.01, 0.05]
testing(x_train, y_train,x_val, y_val, alphas, thetass, w0, 400)
testing(x_train, y_train,x_val, y_val, alphas, thetass,w0, 1000)