import numpy as np
from matplotlib import pyplot as plt
from regressions import return_full_data, standarize, split_data

import copy

import copy


class Regressor(object):
    def __init__(self, theta, alpha):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        # array tamaño
        self.theta = copy.deepcopy(theta)
        self.alpha = copy.deepcopy(alpha)

    def cost(self, x, y):
        # funcio de cost
        m = len(y)
        predictions = x.dot(self.theta)
        # print(predictions)
        errors = np.subtract(predictions, y)
        # print ("Errors: ", errors)
        J = (1 / (2 * m)) * np.sum(np.square(errors))

        return J

    def predict(self, x_test, y_test):
        # implementar aqui la funció de prediccio

        return x_test.T.dot(self.theta)

    def train(self, x, y, iterations: int):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        cost_history = np.zeros(iterations, dtype=object)
        m = len(y)
        for it in range(iterations):
            for j in range(len(self.theta)):
                gradient = 0
                for i in range(m):
                    ##esto dentro de update ---
                    predictions = x[i].dot(self.theta)
                    gradient += (predictions - y[i]) * x[i][j]

                gradient = gradient * (1 / (2 * m))
                self.theta[j] = self.theta[j] - (self.alpha * gradient)
            cost_history[it] = self.cost(x, y)
        return self.theta, cost_history


# Obtenim les dades netes i les estandaritzem
full_data = return_full_data()
standarized_data = standarize(full_data)
standarized_data.hist(bins=25)
plt.show()
standarized_data = standarized_data.drop('wind_direction_corr',axis=1)

x_train, y_train, x_val, y_val = split_data(standarized_data.values[:, :5], standarized_data.values[:, 5])

# Definim una funcio per a testejar el nostre descens de gradient
def testing(x_train, y_train, x_val, y_val, alphas, theta, iterations):
    cost_history = []
    meanr2s = []
    theta1 = 0
    for j in range(len(alphas)):
        alpha = alphas[j]
        theta1 = copy.deepcopy(theta)
        regr = Regressor(theta1, alpha)
        # y_train = np.zeros(4)
        theta1, cost = regr.train(x_train, y_train, iterations)
        cost_history.append(cost)
        print('Final value of theta =', theta1)
        print('First 5 values from cost_history =', cost[:5])
        print('Last 5 values from cost_history =', cost[-5:])

        acc = 0
        for i in range(len(y_val)):
            y_pred = regr.predict(x_val[i], y_val[i])
            rss = np.sum((y_pred - y_val[i]) ** 2)
            tss = score = np.sum((y - y.mean()) ** 2)
            r2score = 1 - (rss / tss)
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

    plt.figure(2)
    plt.bar(['0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05'], meanr2s)
    plt.grid()
    plt.ylabel('meanr2 score(de 0.9999 a 1.00)')
    plt.xlabel('alpha value')
    plt.ylim([0.9999, 1.0])
    plt.title("r2 score mitjà per learning rate - {} iteracions".format(iterations), y=1.08)
    return 0
#Definim els valors per alpha i theta (conjunt de pesos) per fer el descens de gradient
thetass = np.random.uniform(0.0, 1.0, size=5)
alphas = [0.0001, 0.0005, 0.001,0.005,0.01, 0.05]


testing(x_train, y_train,x_val, y_val,alphas, thetass, 400)

testing(x_train, y_train,x_val, y_val, alphas, thetass, 1000)

