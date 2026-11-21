import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__( 
        self, num_classes: int = 1000, dropout: float = 0.7
    ) -> None:
        super().__init__() # YOUR CODE HERE     

        self.features = nn.Sequential( # Define a CNN architecture. Remember to use the variable num_classes
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # to size appropriately the output of your classifier, and if you use
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # the Dropout layer, use the variable "dropout" to indicate how much
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # to use (like nn.Dropout(p=dropout))
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x) # YOUR CODE HERE: process the input tensor through the
        x = self.pool(x) # feature extractor, the pooling and the final linear
        x = self.classifier(x) # layers (if appropriate for the architecture chosen)
        
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers=0) #return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"