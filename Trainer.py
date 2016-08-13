from scipy import optimize


class Trainer:

    def __init__(self, network):
        self.network = network
        self.i = 0

    def callback_f(self, params):
        self.network.set_params(params)
        cost = self.network.cost_function(self.x, self.y)
        self.J.append(cost)
        # print(str(self.i)+"th Iteration Over: COst:"+str(cost))
        self.i += 1

    def cost_function_wrapper(self, params, x, y):
        self.network.set_params(params)
        return self.network.costs(x, y)

    def train(self, x, y):
        # Make an internal variable for the callback function:
        self.x = x
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.network.get_params()

        options = {'maxiter': 100, 'disp': False}
        _res = optimize.minimize(self.cost_function_wrapper, params0, jac=True, method='CG', \
                                 args=(x, y), options=options, callback=self.callback_f)
        print("Finished")
        self.network.set_params(_res.x)
