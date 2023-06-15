import torch
import torchvision.models as models


# Save model weights
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# Load model weights
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# Save entire model
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model, 'model.pth')

# Load entire model
model = torch.load('model.pth')
