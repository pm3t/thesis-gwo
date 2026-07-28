import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, n_particles, max_iter, dim=3, lb=0, ub=1, w=0.729, c1=1.494, c2=1.494):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient
        self.c2 = c2    # Social coefficient
        
        # Initialize positions (random within [lb, ub] and normalized)
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        for i in range(self.n_particles):
            self.positions[i] = self.positions[i] / np.sum(self.positions[i])
            
        # Initialize velocities to zero
        self.velocities = np.zeros((self.n_particles, self.dim))
        
        # Personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, float("inf"))
        
        # Global best
        self.gbest_position = np.zeros(self.dim)
        self.gbest_score = float("inf")

    def fitness_function(self, weights, y_true, y_pred_ma, y_pred_es, y_pred_lr):
        """
        Calculate fitness (MAPE) for a given set of weights.
        """
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
        positions_history = []  # formatted identical to GWO for UI compatibility
        
        for t in range(self.max_iter):
            # Calculate fitness for all particles
            for i in range(self.n_particles):
                # Clip position to boundary
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                if np.sum(self.positions[i]) > 0:
                    self.positions[i] = self.positions[i] / np.sum(self.positions[i])
                else:
                    self.positions[i] = np.random.dirichlet(np.ones(self.dim))

                fitness = self.fitness_function(self.positions[i], y_true, y_pred_ma, y_pred_es, y_pred_lr)
                
                # Update Personal Best
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                # Update Global Best
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i].copy()
            
            # Update velocities and positions
            # Inertia weight damping (linear decrease from w_max to w_min)
            w_t = self.w - (t / self.max_iter) * (self.w - 0.4)
            
            for i in range(self.n_particles):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # Cognitive & Social components
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                
                # Update velocity
                self.velocities[i] = w_t * self.velocities[i] + cognitive + social
                
                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Normalize position to sum up to 1
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                if np.sum(self.positions[i]) > 0:
                    self.positions[i] = self.positions[i] / np.sum(self.positions[i])
                else:
                    self.positions[i] = np.random.dirichlet(np.ones(self.dim))

            convergence_curve.append(self.gbest_score)
            
            # Save snapshot compatible with GWO visualization format
            # we map alpha -> gbest, beta -> pbest average, delta -> worst particle (or dummy)
            positions_history.append({
                'wolves': self.positions.copy(),
                'alpha': self.gbest_position.copy(),
                'beta': self.gbest_position.copy(),  # fallback to gbest to reuse GWO visualization
                'delta': self.gbest_position.copy(), # fallback to gbest
                'alpha_score': self.gbest_score,
            })
            
        # Ensure final global best position is normalized
        best_weights = self.gbest_position / np.sum(self.gbest_position)
        return best_weights, convergence_curve, positions_history
