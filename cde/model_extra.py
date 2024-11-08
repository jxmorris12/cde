from cde.model import BiEncoder
import torch
import transformers


class UNetTransform(torch.nn.Module):
    def __init__(
            self,
            src_dim: int,
            target_dim: int,
        ):
        super().__init__()
        
        import diffusers
        self.base = diffusers.UNet2DModel(
            sample_size=(32, 32), # the target image resolution
            in_channels=1, # the number of input channels, 3 for RGB images
            out_channels=1, # the number of output channels
            layers_per_block=2, # how many ResNet layers to use per UNet block
            block_out_channels=(128,128,256,256,512,512), # the number of output channels for eaxh UNet block
            down_block_types=(
                "DownBlock2D", # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", # a regular ResNet upsampling block
                "AttnUpBlock2D", # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.internal_dim = src_dim
        self.reshape_factor = 32
        assert self.internal_dim % self.reshape_factor == 0
        self.in_projection = torch.nn.Linear(src_dim, 1024)
        self.out_projection = torch.nn.Linear(1024, target_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.in_projection(x)
        x = x[:, None, None, :].reshape(-1, 1, 32, 32) # repeat embedding into an image
        output = self.base(x, timestep=0)
        z = output.sample.reshape(batch_size, 1024).contiguous()
        return self.out_projection(z)

class BiEncoderPlusPlus(BiEncoder):
    embedder: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
        ):
        super().__init__(config=config)
        self.unet = UNetTransform(
            src_dim=self.hidden_size,
            target_dim=self.hidden_size,
        )

    def forward(
            self, *args, **kwargs
        ) -> torch.Tensor:
        output = super().forward(*args, **kwargs)
        return self.unet(output)