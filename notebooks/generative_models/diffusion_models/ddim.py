import torch
import torch.nn as nn

from ddpm import DDPM


class DDIM(DDPM):
    """
    DDPM Sampling.

    Args:
        eps_model: A neural network model that predicts the noise term given a tensor.
        betas: A tuple containing two floats, which are parameters used in the DDPM schedule.
        eta: Scaling factor for the random noise term.
        n_timesteps: Mumber of timesteps in the diffusion process.
        criterion: Loss function.
    """

    def __init__(
            self,
            eps_model: nn.Module,
            betas: tuple[float, float],
            eta: float,
            n_timesteps: int,
            criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDIM, self).__init__(eps_model, betas, n_timesteps, criterion)
        self.eta = eta

    def sample(self, n_samples: int, size: torch.Tensor, device: str) -> torch.Tensor:
        # Initialize x_i with random noise from a standard normal distribution
        # x_i corresponds to x_T in the diffusion process, where T is the total number of timesteps
        x_i = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1)

        # Iterate backwards through the timesteps from n_timesteps to 1
        for i in range(self.n_timesteps, 1, -1):
            # Sample additional random noise z, unless i is 1 (in which case z is 0, i.e., no additional noise)
            z = torch.randn(n_samples, *size).to(device) if i > 1 else 0  # z ~ N(0, 1) for i > 1, else z = 0

            # Predict the noise eps to be removed at the current timestep, using the eps_model
            # The current timestep i is normalized by n_timesteps and replicated for each sample
            eps = self.eps_model(x_i, torch.tensor(i / self.n_timesteps).to(device).repeat(n_samples, 1))

            # Calculate the predicted x0 (original data) at timestep 'i'
            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()

            # Compute coefficients for the DDIM sampling process.
            c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            # Update x_i using the DDIM formula.
            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i
