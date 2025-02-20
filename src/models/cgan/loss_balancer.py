import torch


class PastGradientLossBalancer:
    def __init__(self, loss_names, smoothing=0.9, initial_weights=None, intensities=None):
        self.loss_names = loss_names
        self.smoothing = smoothing
        self.initial_weights = initial_weights
        self.loss_history = {name: [] for name in loss_names}  # Store past losses
        self.smoothed_weights = {name: 1.0 for name in loss_names}  # Store smoothed losses

        # Initialize weights with provided initial weights or default to 1.0
        if initial_weights is not None:
            self.loss_weights = {name: initial_weights.get(name, 1.0) for name in loss_names}
        else:
            self.loss_weights = {name: 1.0 for name in loss_names}

        # Initialize intensities with provided values or default to 1.0
        if intensities is not None:
            self.intensities = {name: intensities.get(name, 1.0) for name in loss_names}
        else:
            self.intensities = {name: 1.0 for name in loss_names}

    def estimate_gradients(self):
        """
        Estimate gradients based on past loss values (finite differences).
        """
        estimated_grads = {}
        for name, history in self.loss_history.items():
            if len(history) > 2:
                estimated_grads[name] = (history[-1] - history[-2])  # Change in loss
            else:
                estimated_grads[name] = 1.0
            
        return estimated_grads

    def update_weights(self):
        """
        Adjust loss weights inversely proportional to estimated gradients.
        """
        estimated_grads = self.estimate_gradients()
        
        grad_range = max(estimated_grads.values()) - min(estimated_grads.values())
        
        for name, history in self.loss_history.items():
            if len(history) > 2:
                self.loss_weights[name] = (estimated_grads[name] - min(estimated_grads.values()))/ (grad_range + 1e-8)
            else:
                self.loss_weights[name] = self.initial_weights[name]

    def get_loss_weights(self, loss_values):
        """
        Update history and compute loss weights.
        """
        
    # Update the loss history
        for name in self.loss_names:
            self.loss_history[name].append(loss_values[name])

        # Update the weights based on the estimated gradients
        self.update_weights()
        # Smooth the weights using exponential moving average
        for name in self.loss_names:
            self.smoothed_weights[name] = self.smoothing * self.smoothed_weights[name] + (1 - self.smoothing) * self.loss_weights[name]
            
        weights = {name: self.smoothed_weights[name] * self.intensities[name] for name in self.loss_names}
        return weights
