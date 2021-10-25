import numpy as np
from matplotlib import pyplot as plt
from regressions import return_full_data, standarize, split_data

class Regressor(object):
    def __init__(self, theta, alpha):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        # array tamaño
        self.theta = theta
        self.alpha = alpha

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

    def __update(self, hy, y):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real.

        pass

    def train(self, x, y, iterations: int, epsilon):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        cost_history = np.zeros(iterations, dtype=object)
        m = len(y)
        for it in range(iterations):
            for j in range(len(theta)):
                gradient = 0
                for i in range(m):
                    ##esto dentro de update ---
                    predictions = x[i].dot(theta)
                    gradient += (predictions - y[i]) * x[i][j]

                gradient = gradient * (1 / (2 * m))
                self.theta[j] = self.theta[j] - (alpha * gradient)
            cost_history[it] = self.cost(x, y)
        return self.theta, cost_history


# Obtenim les dades netes i les estandaritzem
full_data = return_full_data()
standarized_data = standarize(full_data)
standarized_data.hist(bins=25)
plt.show()
standarized_data = standarized_data.drop('wind_direction_corr',axis=1)

x_train, y_train, x_val, y_val = split_data(standarized_data.values[:, :5], standarized_data.values[:, 5])

# Definim els valors per alpha i theta (conjunt de pesos) per fer el descens de gradient
alpha = 0.0001
theta = np.random.uniform(0.0, 1.0, size=5)
iterations = 400
regr = Regressor(theta,alpha)
np.set_printoptions(precision=4)

# Fem el descens de gradient
theta, cost_history = regr.train(x_train, y_train, iterations, 0 )
print('Final value of theta =', theta)
print('First 5 values from cost_history =', cost_history[:5])
print('Last 5 values from cost_history =', cost_history[-5 :])

# Avaluem el resultat
acc = 0
for i in range (len(y_val)):
    y_pred = regr.predict(x_val[i],y_val[i])
    rss = np.sum((y_pred - y_val[i]) ** 2)
    tss = score = np.sum((y-y.mean()) ** 2)
    r2score = 1 - (rss/tss)
    acc = acc + r2score
meanr2 = acc/len(y_val)
print(meanr2)

# Provem amb alpha = 0.001
alpha = 0.001
theta, cost_history = regr.train(x_train, y_train, iterations, 0 )
print('Final value of theta =', theta)
print('First 5 values from cost_history =', cost_history[:5])
print('Last 5 values from cost_history =', cost_history[-5 :])
acc = 0
for i in range (len(y_val)):
    y_pred = regr.predict(x_val[i],y_val[i])
    rss = np.sum((y_pred - y_val[i]) ** 2)
    tss = score = np.sum((y-y.mean()) ** 2)
    r2score = 1 - (rss/tss)
    acc = acc + r2score
meanr2 = acc/len(y_val)
print(meanr2)

# Provem amb alpha 0.01
alpha = 0.01
#y_train = np.zeros(4)
theta, cost_history = regr.train(x_train, y_train, iterations, 0 )
print('Final value of theta =', theta)
print('First 5 values from cost_history =', cost_history[:5])
print('Last 5 values from cost_history =', cost_history[-5 :])
for i in range (len(y_val)):
    y_pred = regr.predict(x_val[i],y_val[i])
    rss = np.sum((y_pred - y_val[i]) ** 2)
    tss = score = np.sum((y-y.mean()) ** 2)
    r2score = 1 - (rss/tss)
    acc = acc + r2score
meanr2 = acc/len(y_val)
print(meanr2)

