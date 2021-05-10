import numpy as np

class AttackWrapper:

    def attack_feasibility(self, X, y, order=np.inf, epsilon=0.0, options={}):
        X_distances = self.attack_distance(X, y, order, options)
        return X_distances < epsilon
        
    def attack_distance(self, X, y, order=np.inf, options={}):
        X_adv = self.adversarial_examples(X, y, order, options)
        return np.linalg.norm(X - X_adv, ord=order, axis=1)

    def adversarial_examples(self, X, y, order=np.inf, options={}):
        raise NotImplementedError
