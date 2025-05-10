from torch import nn

'''
class ConditionalBatchNorm2d(nn.Module):
    """https://github.com/pytorch/pytorch/issues/8985"""

    def __init__(self, num_features, num_classes, act=None):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        self.act = act

    def forward(self, x, y, class_prob=1.0):
        out = self.bn(x)

        gamma, beta = self.embed(y).chunk(2, 1)
        #out = class_prob.view(-1, 1, 1, 1).expand_as(gamma.view(-1, self.num_features, 1, 1)) * gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        if self.act is not None:
            out = self.act(out)
        return out
'''

class ConditionalBatchNorm2d(nn.Module):
    """Conditional BatchNorm using classifier output"""
    
    def __init__(self, num_features, num_classes,act=None, classifier_output_dim=1280, use_signal_from_classifier = False):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.use_signal_from_classifier = use_signal_from_classifier
        
        if self.use_signal_from_classifier:
            # Linear layer to map classifier output to the correct feature size
            self.fc = nn.Linear(classifier_output_dim, num_features * 2)
        else:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
            
        
        # Optional activation function
        self.act = act

    def forward(self, x, y, class_prob=1.0, classifier_output=None):
        # Apply batch normalization
        if self.use_signal_from_classifier:
            out = self.bn(x)
            
            # Pass classifier output through the linear layer to get gamma and beta
            gamma_beta = self.fc(classifier_output)  # Shape: (batch_size, num_features * 2)
            
            # Split the output into gamma and beta
            gamma, beta = gamma_beta.chunk(2, 1)
            
            # Apply gamma and beta (scaling and shifting)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            out = self.bn(x)
            gamma, beta = self.embed(y).chunk(2, 1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        
        if self.act is not None:
            out = self.act(out)
        
        return out