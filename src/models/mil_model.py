import torch
from torch import nn
from torchvision.models import resnet34

class AttentionModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        return (attention_weights * x).sum(dim=1)

class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet34(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the final FC layer
        self.attention = AttentionModule(512, 256)  # Adjust these dimensions as needed
        self.classifier = nn.Sequential(
            nn.Linear(512 + 3, 256),  # 2048 from attention, 3 from additional features
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, additional_features):
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(-1, channels, height, width)  # Reshape to collapse batch_size and num_images

        reshaped_features = self.feature_extractor(reshaped_images)
        image_features = reshaped_features.view(batch_size, num_images, -1)  # Reshape to separate batch_size and num_images

        image_representation = self.attention(image_features)
        image_representation = image_representation.float()
        additional_features = additional_features.float()
        final_representation = torch.cat([image_representation, additional_features], dim=1)
        return self.classifier(final_representation)