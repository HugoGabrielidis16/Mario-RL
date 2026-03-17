import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for efficient exploration (NoisyNet)
    Replaces epsilon-greedy exploration with learnable noise parameters
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise buffers"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Generate scaled noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class ImpalaResidualBlock(nn.Module):
    """
    Residual block from IMPALA architecture
    Optimized for RL with efficient computation
    """
    def __init__(self, channels):
        super(ImpalaResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class ImpalaBlock(nn.Module):
    """
    Full IMPALA block: Conv -> MaxPool -> 2x Residual Blocks
    """
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ImpalaResidualBlock(out_channels)
        self.res_block2 = ImpalaResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class ImpalaDQN(nn.Module):
    """
    IMPALA-style CNN with DQN head
    Designed for fast convergence in RL tasks with frame stacking

    Args:
        state_shape: Tuple of (frame_stack, height, width)
        n_actions: Number of possible actions
        channels: List of channel sizes for each IMPALA block (default: [16, 32, 32])
        dueling: Whether to use Dueling DQN architecture
        noisy: Whether to use Noisy layers for exploration
    """
    def __init__(self, state_shape, n_actions, channels=[16, 32, 32],
                 dueling=True, noisy=False):
        super(ImpalaDQN, self).__init__()

        # Parse state shape
        if len(state_shape) == 3:
            self.frame_stack, self.height, self.width = state_shape
        else:
            raise ValueError(f"Expected state_shape (frame_stack, H, W), got {state_shape}")

        self.n_actions = n_actions
        self.dueling = dueling
        self.noisy = noisy

        print(f"🧠 Initializing ImpalaDQN:")
        print(f"   • Frame stack: {self.frame_stack}")
        print(f"   • Input size: {self.height}x{self.width}")
        print(f"   • Channels: {channels}")
        print(f"   • Actions: {n_actions}")
        print(f"   • Dueling: {dueling}")
        print(f"   • Noisy: {noisy}")

        # IMPALA CNN backbone
        self.blocks = nn.ModuleList()
        in_channels = self.frame_stack

        for out_channels in channels:
            self.blocks.append(ImpalaBlock(in_channels, out_channels))
            in_channels = out_channels

        # Calculate feature size after conv blocks
        self.feature_size = self._get_conv_output_size()

        # Dueling DQN architecture
        if dueling:
            if noisy:
                # Value stream
                self.value_fc1 = NoisyLinear(self.feature_size, 512)
                self.value_fc2 = NoisyLinear(512, 1)

                # Advantage stream
                self.advantage_fc1 = NoisyLinear(self.feature_size, 512)
                self.advantage_fc2 = NoisyLinear(512, n_actions)
            else:
                # Value stream
                self.value_fc1 = nn.Linear(self.feature_size, 512)
                self.value_fc2 = nn.Linear(512, 1)

                # Advantage stream
                self.advantage_fc1 = nn.Linear(self.feature_size, 512)
                self.advantage_fc2 = nn.Linear(512, n_actions)
        else:
            # Standard DQN head
            if noisy:
                self.fc1 = NoisyLinear(self.feature_size, 512)
                self.fc2 = NoisyLinear(512, n_actions)
            else:
                self.fc1 = nn.Linear(self.feature_size, 512)
                self.fc2 = nn.Linear(512, n_actions)

        # Initialize weights
        self._initialize_weights()

        # Print model info
        param_count = sum(p.numel() for p in self.parameters())
        print(f"   • Parameters: {param_count:,}")
        print(f"   • Feature size: {self.feature_size}")

    def _get_conv_output_size(self):
        """Calculate the output size after conv blocks"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.frame_stack, self.height, self.width)
            x = dummy_input
            for block in self.blocks:
                x = block(x)
            return int(torch.prod(torch.tensor(x.shape[1:])))

    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization for RL"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, frame_stack, height, width)

        Returns:
            Q-values of shape (batch_size, n_actions)
        """
        # IMPALA CNN backbone
        for block in self.blocks:
            x = block(x)

        # Flatten features
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        # Dueling or standard DQN head
        if self.dueling:
            # Value stream
            value = F.relu(self.value_fc1(x))
            value = self.value_fc2(value)

            # Advantage stream
            advantage = F.relu(self.advantage_fc1(x))
            advantage = self.advantage_fc2(advantage)

            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)

        return q_values

    def reset_noise(self):
        """Reset noise in NoisyLinear layers (call before each forward pass during training)"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class ImpalaDuelingDQN(ImpalaDQN):
    """
    Convenience class for IMPALA CNN with Dueling architecture
    """
    def __init__(self, state_shape, n_actions, channels=[16, 32, 32]):
        super().__init__(state_shape, n_actions, channels=channels,
                        dueling=True, noisy=False)


class ImpalaNoisyDQN(ImpalaDQN):
    """
    Convenience class for IMPALA CNN with Noisy layers
    """
    def __init__(self, state_shape, n_actions, channels=[16, 32, 32]):
        super().__init__(state_shape, n_actions, channels=channels,
                        dueling=False, noisy=True)


class ImpalaDuelingNoisyDQN(ImpalaDQN):
    """
    Convenience class for IMPALA CNN with both Dueling and Noisy layers
    Full Rainbow DQN configuration (without distributional RL)
    """
    def __init__(self, state_shape, n_actions, channels=[16, 32, 32]):
        super().__init__(state_shape, n_actions, channels=channels,
                        dueling=True, noisy=True)


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing ImpalaDQN architectures\n")

    state_shape = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    n_actions = 12
    batch_size = 32

    # Test all variants
    variants = [
        ("Standard IMPALA-DQN", ImpalaDQN(state_shape, n_actions, dueling=False, noisy=False)),
        ("IMPALA + Dueling", ImpalaDuelingDQN(state_shape, n_actions)),
        ("IMPALA + Noisy", ImpalaNoisyDQN(state_shape, n_actions)),
        ("IMPALA + Dueling + Noisy", ImpalaDuelingNoisyDQN(state_shape, n_actions)),
    ]

    for name, model in variants:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        # Test forward pass
        sample_input = torch.randn(batch_size, *state_shape)
        output = model(sample_input)

        print(f"✅ Input shape: {sample_input.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test gradient flow
        loss = output.sum()
        loss.backward()
        print(f"✅ Gradient test: Passed")

        # Test noise reset for noisy models
        if hasattr(model, 'noisy') and model.noisy:
            model.reset_noise()
            print(f"✅ Noise reset: Passed")

    print(f"\n{'='*60}")
    print("🎉 All tests passed!")
    print(f"{'='*60}")
