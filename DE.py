import numpy as np

class DE:

    '''Differential Evolution for single objective optimization problem'''

    def __init__(self, problem, n_dim=10, n_gen=100, n_pop=10, ub=-100, lb=100, F=0.8, CR=0.5):
        self.problem = problem
        self.n_dim = n_dim
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.ub = ub
        self.lb = lb
        self.F = F
        self.CR = CR

    def optimize(self):
       
        # initialization
        x = self.lb + (self.ub - self.lb) * np.random.rand(self.n_pop, self.n_dim)
        f_x = np.zeros((self.n_pop,1))
        f_u = np.zeros((self.n_pop,1))
        
        for i in range(self.n_pop):
            f_x[i] = self.problem.evaluate(x[i])

        gen = 0
        
        # iteration
        while gen < self.n_gen:

            # mutation
            random_idx = np.random.randint(0, self.n_pop, size=(self.n_pop, 3))
            r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
            # v[i]=x[r1]+F(x[r2]-x[r3])
            v = x[r1, :] + self.F * (x[r2, :] - x[r3, :])

            # repair
            mask_bound = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_pop, self.n_dim))
            v = np.where(v < self.lb, mask_bound, v)
            v = np.where(v > self.ub, mask_bound, v)

            # crossover
            mask_co = (np.random.rand(self.n_pop, self.n_dim) < self.CR)
            jrand =np.random.randint(0, self.n_dim, size=(1, self.n_pop))
            for i in range(self.n_pop): 
                mask_co[i,jrand[0,i]] = True
            u = np.where(mask_co, v, x)

            # selection
            for i in range(self.n_pop):
                f_u[i] = self.problem.evaluate(u[i])

            x = np.where(f_u<f_x, u, x)
            f_x = np.where(f_u<f_x, f_u, f_x)

            gen = gen + 1

        return np.min(f_x)
       