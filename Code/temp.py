import torch
import torch.nn.functional as F
from torch import cuda

# Assuming 'output' is the output of the final layer
output = torch.randn(1, 1000)  # Example tensor

# Apply the softmax operation
probabilities = F.softmax(output, dim=1)

# 'probabilities' now contains the class probabilities
print(cuda.is_available() )
