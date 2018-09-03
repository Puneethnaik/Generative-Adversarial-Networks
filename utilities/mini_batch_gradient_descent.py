import numpy as np

#This is crafted especially for normal distribution for MLE.
class GradientDescentOptimizer:
    def __init__(self, X, tolerance, learning_rate):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.X = X
        if(len(X.shape) == 1):
            self.number_of_points = X.shape[0]
            self.number_of_variables = 2
        else:
            self.number_of_points, self.number_of_variables = X.shape
        # self.number_of_variables -= 1 #we subtract the extra bias term
        print(self.number_of_points, self.number_of_variables, "hello")

    def optimize(self):
        self.theta = np.array([np.random.randint(1, 10) for _ in range(self.number_of_variables)]) #we choose a random value for theta
        self.theta = np.resize(self.theta, new_shape=(self.number_of_variables, 1))
        self.theta = self.theta.astype(float)
        prev_value = 1
        current_value = 2
        print("theta assigned", self.theta)
        print("X", self.X)
        while abs(prev_value - current_value) >= self.tolerance:
            gradient = self.theta.copy()
            for i in range(2):
                if i == 0:
                    gradient[i][0] = self.learning_rate * (1.0 / self.number_of_points) * np.sum((self.X - self.theta[0]))
                else :
                    gradient[i][0] = self.learning_rate * (1.0 / self.number_of_points) * np.sum((-1.0 / (2.0 * self.theta[1])) + ((self.X - self.theta[0]) ** 2 / (2.0 * (self.theta[1]) ** 2)))
            # print("gradient ", gradient)
            if self.theta[1] + gradient[1][0] < 0:
                break
            self.theta = self.theta + gradient
            prev_value = current_value
            current_value = np.sum(-np.log(np.sqrt(2 * np.pi)) - np.log(np.sqrt(self.theta[1])) - ((self.X - self.theta[0]) ** 2 / (2 * self.theta[1])))
            print("loss function " + str(current_value))
            print("theta ", self.theta)
        return self.theta