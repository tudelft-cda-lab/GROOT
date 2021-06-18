import numpy as np


class AttackWrapper:
    """
    Wrapper for adversarial attack algorithms. Attack implementations need to
    define an `adversarial_examples()`, then `attack_distance()` and `attack_feasibility()`
    will be handled by this base wrapper. Some attacks can implement `attack_feasibility()`
    and `attack_distance()` more efficiently though, in which case they can be overriden.
    """

    def attack_feasibility(self, X, y, order=np.inf, epsilon=0.0, options={}):
        """
        Determine whether an adversarial example is feasible for each sample given the maximum perturbation radius epsilon.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to attack.
        y : array-like of shape (n_samples,)
            True labels for the samples.
        order : {0, 1, 2, inf}, optional
            L-norm order to use. See numpy documentation of more explanation.
        epsilon : float, optional
            Maximum distance by which samples can move.
        options : dict, optional
            Extra attack-specific options.

        Returns
        -------
        ndarray of shape (n_samples,) of booleans
            Vector of True/False. Whether an adversarial example is feasible.
        """
        X_distances = self.attack_distance(X, y, order, options)
        return X_distances < epsilon

    def attack_distance(self, X, y, order=np.inf, options={}):
        """
        Determine the perturbation distance for each sample to make an adversarial example.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to attack.
        y : array-like of shape (n_samples,)
            True labels for the samples.
        order : {0, 1, 2, inf}, optional
            L-norm order to use. See numpy documentation of more explanation.
        options : dict, optional
            Extra attack-specific options.

        Returns
        -------
        ndarray of shape (n_samples,) of floats
            Distances to create adversarial examples.
        """
        X_adv = self.adversarial_examples(X, y, order, options)
        return np.linalg.norm(X - X_adv, ord=order, axis=1)

    def adversarial_examples(self, X, y, order=np.inf, options={}):
        """
        Create adversarial examples for each input sample. This method has to be overriden!

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to attack.
        y : array-like of shape (n_samples,)
            True labels for the samples.
        order : {0, 1, 2, inf}, optional
            L-norm order to use. See numpy documentation of more explanation.
        options : dict, optional
            Extra attack-specific options.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Adversarial examples.
        """
        raise NotImplementedError
