# src/model_architecture.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class PlantDiseaseModel(nn.Module):
    """
    A custom model for plant disease detection using a pre-trained EfficientNetV2 backbone
    with a custom classifier head.
    """
    def __init__(self, config, num_classes):
        super(PlantDiseaseModel, self).__init__()

        # Load pre-trained EfficientNetV2B3 model
        self.base_model = models.efficientnet_v2_s(pretrained=True)  # Using EfficientNetV2-S which is similar to B3

        # Freeze all layers initially for transfer learning phase
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get the number of features from the last layer of the base model
        num_features = self.base_model.classifier[1].in_features

        # Replace the classifier head with our custom architecture
        self.dropout1 = nn.Dropout(p=config['dropout_1'])
        self.dense1 = nn.Linear(num_features, config['dense_1'])
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=config['dropout_2'])
        self.output_layer = nn.Linear(config['dense_1'], num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Replace the base model's classifier
        self.base_model.classifier = nn.Identity()  # Remove original classifier

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def build_model(config, num_classes):
    """
    Builds the model based on the configuration (initially frozen for transfer learning).
    """
    model = PlantDiseaseModel(config, num_classes)

    print(f"Model built successfully with {num_classes} classes.")

    # Print model summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Try to print a summary (requires torchsummary package)
    try:
        # Assuming input shape is (batch_size, channels, height, width)
        input_shape = tuple(config['input_shape'])  # [300, 300, 3] -> (3, 300, 300)
        # Reorder to (C, H, W) for PyTorch
        input_shape_pytorch = (input_shape[2], input_shape[0], input_shape[1])
        summary(model, input_shape_pytorch)
    except ImportError:
        print("torchsummary not available, skipping model summary")
    except Exception as e:
        print(f"Error generating model summary: {e}")

    return model


def compile_model_for_transfer_learning(model, config):
    """
    Prepares and compiles the model for the first phase (transfer learning - head training).
    """
    # The base model is already frozen in the constructor
    # Only the new classifier layers are trainable at this point

    # Define optimizer with the learning rate for head training
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_head']
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    print("Model compiled for Transfer Learning (Head Training).")
    print(f"Number of parameters to train: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model, optimizer, criterion


def compile_model_for_fine_tuning(model, config):
    """
    Prepares and compiles the model for the second phase (fine-tuning).
    """
    # Unfreeze the base model layers for fine-tuning
    for param in model.base_model.parameters():
        param.requires_grad = True

    # Optionally freeze lower layers based on config
    fine_tune_at = config['fine_tune_at_layer']
    if fine_tune_at > 0:
        print(f"Freezing all layers except the top {fine_tune_at} layers.")
        # This is more complex in PyTorch - we'll need to control this differently
        # For now, we'll unfreeze all and let the learning rate handle it
    else:
        print("Unfreezing all layers for full fine-tuning.")

    # Define optimizer with a lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr_fine_tune']
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    print("Model compiled for Fine-Tuning.")
    print(f"Number of parameters to train: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model, optimizer, criterion