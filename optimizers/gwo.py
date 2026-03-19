import numpy as np

class GreyWolfOptimizer:
    def __init__(self, n_wolves, max_iter, dim=3, lb=0, ub=1):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        
        # Initialize wolf positions (random within [lb, ub])
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))
        
        # Normalize initial positions so they sum up to 1
        for i in range(self.n_wolves):
            self.positions[i] = self.positions[i] / np.sum(self.positions[i])
            
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")

    def fitness_function(self, weights, y_true, y_pred_ma, y_pred_es, y_pred_lr):
        """
        Calculate fitness (MAPE) for a given set of weights.
        """
        # Ensure weights sum to 1
        weights = np.array(weights)
        if np.sum(weights) == 0:
            return float("inf")
        
        weights = weights / np.sum(weights)
        
        # Ensemble prediction
        y_ensemble = (weights[0] * y_pred_ma + 
                     weights[1] * y_pred_es + 
                     weights[2] * y_pred_lr)
        
        # MAPE calculation
        y_true, y_ensemble = np.array(y_true), np.array(y_ensemble)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_ensemble[mask]) / y_true[mask])) * 100
        
        return mape

    def optimize(self, y_true, y_pred_ma, y_pred_es, y_pred_lr):
        convergence_curve = []
        
        for t in range(self.max_iter):
            for i in range(self.n_wolves):
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # Calculate fitness
                fitness = self.fitness_function(self.positions[i], y_true, y_pred_ma, y_pred_es, y_pred_lr)
                
                # Update Alpha, Beta, Delta
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            # Parameter 'a' decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)
            
            # Update positions
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # For Alpha
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # For Beta
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # For Delta
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # New position
                    self.positions[i, j] = (X1 + X2 + X3) / 3
                
                # Normalize new position so weights sum to 1
                if np.sum(self.positions[i]) > 0:
                    self.positions[i] = self.positions[i] / np.sum(self.positions[i])
                else:
                    # In case of zeros, re-initialize randomly
                    self.positions[i] = np.random.dirichlet(np.ones(self.dim))

            convergence_curve.append(self.alpha_score)
            
        # Ensure final alpha position is normalized
        best_weights = self.alpha_pos / np.sum(self.alpha_pos)
        return best_weights, convergence_curve
