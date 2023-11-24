import torch
import torch.nn as nn


class DDPM(nn.Module):
    """
    DDPM Sampling.

    Args:
        eps_model: A neural network model that predicts the noise term given a tensor.
        betas: A tuple containing two floats, which are parameters used in the DDPM schedule.
        n_timesteps: Mumber of timesteps in the diffusion process.
        criterion: Loss function.
    """

    def __init__(
            self,
            eps_model: nn.Module,
            betas: tuple[float, float],
            n_timesteps: int,
            criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # Register_buffer allows us to freely access these tensors by name.
        for k, v in ddpm_schedules(betas[0], betas[1], n_timesteps).items():
            self.register_buffer(k, v)

        self.n_timesteps = n_timesteps
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward diffusion process (akin to Algorithm 1 from the paper).
        The goal is to predict the noise term (eps_sample) from a tensor x_t using the model eps_model,
        and compute the loss between the true noise (eps_sample) and the predicted noise.

        Steps:
        1. Randomly samples a timestep `timestep_sample` from a uniform distribution between 1 and n_timesteps.
        2. Samples a noise term `eps_sample` from a standard normal distribution.
        3. Computes x_t, which is the tensor after adding the noise, using the formula:
           x_t = sqrt(alphabar) * x_0 + sqrt(1 - alphabar) * eps_sample

        Parameters:
            x: The input tensor to the diffusion process.

        Returns:
            The computed loss between the true noise and the predicted noise.
        """
        # Randomly sample a timestep from a uniform distribution between 1 and n_timesteps
        # The sampled timestep is then transferred to the same device as the input tensor x
        timestep_sample = torch.randint(1, self.n_timesteps + 1, (x.shape[0],)).to(x.device)  # t ~ Uniform(1, n_T)

        # Sample a noise term from a standard normal distribution
        eps_sample = torch.randn_like(x)  # eps ~ N(0, 1)

        # Compute x_t using the formula: x_t = sqrt(alphabar) * x_0 + sqrt(1 - alphabar) * eps_sample
        x_t = (self.sqrtab[timestep_sample, None, None, None] * x +
               self.sqrtmab[timestep_sample, None, None, None] * eps_sample)  # Compute x_t

        # Predict the noise term from x_t using eps_model, and normalize the timestep_sample
        eps_prediction = self.eps_model(x_t, timestep_sample / self.n_timesteps)
        # Compute the loss between the true noise (eps_sample) and the predicted noise
        return self.criterion(eps_sample, eps_prediction)

    def sample(self, n_samples: int, size: torch.Tensor, device: str) -> torch.Tensor:
        """
        Implements the sampling process to generate samples by reversing the diffusion process
        (akin to Algorithm 2 from the paper).
        s
        It starts with an initial noise tensor, and iteratively refines it by removing predicted
        noise and injecting additional noise, over a series of timesteps from `n_timesteps` to 1.

        Args:
            n_samples: The number of samples to generate.
            size: Tensor that defines the size of each sample.
            device: The device on which to perform computations and store tensors (e.g., 'cpu' or 'cuda').

        Returns:
             The generated samples as a tensor of shape (n_samples, *size).
        """
        # Initialize x_i with random noise from a standard normal distribution
        # x_i corresponds to x_T in the diffusion process, where T is the total number of timesteps
        x_i = torch.randn(n_samples, *size).to(device)  # x_T ~ N(0, 1)

        # Iterate backwards through the timesteps from n_timesteps to 1
        for i in range(self.n_timesteps, 0, -1):
            # Sample additional random noise z, unless i is 1 (in which case z is 0, i.e., no additional noise)
            z = torch.randn(n_samples, *size).to(device) if i > 1 else 0  # z ~ N(0, 1) for i > 1, else z = 0

            # Predict the noise eps to be removed at the current timestep, using the eps_model
            # The current timestep i is normalized by n_timesteps and replicated for each sample
            eps = self.eps_model(x_i, torch.tensor(i / self.n_timesteps).to(device).repeat(n_samples, 1))

            # Update x_i by removing the predicted noise eps, optionally adding back in the random noise z
            # This is done according to the reverse process update equation from the DDPM paper
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])  # Remove predicted noise
                    + self.sqrt_beta_t[i] * z
            )

        return x_i


def ddpm_schedules(beta1: float, beta2: float, num_timesteps: int) -> dict[str, torch.Tensor]:
    """
    Computes the DDPM schedules based on the provided betas and number of timesteps.

    This function computes a variety of terms used in a DDPM, based on a linear schedule of beta values
    from `beta1` to `beta2` over `num_timesteps + 1` intervals.

    Args:
        beta1: The starting value of the beta schedule.
        beta2: The ending value of the beta schedule.
        num_timesteps: The total number of timesteps in the diffusion process.

    Returns:
        A dictionary containing the computed terms, each as a tensor:
            - "alpha_t": The alpha schedule, complementary to the beta schedule.
            - "oneover_sqrta": The reciprocal of the square root of the alpha schedule.
            - "sqrt_beta_t": The square root of the beta schedule.
            - "alphabar_t": The cumulative product of the alpha schedule.
            - "sqrtab": The square root of the cumulative product of the alpha schedule.
            - "sqrtmab": The square root of (1 - cumulative product of the alpha schedule).
            - "mab_over_sqrtmab": The ratio of (1 - alpha schedule) to sqrtmab.
    """
    # Ensure beta1 and beta2 are within the open interval (0, 1)
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Compute the beta schedule using linear interpolation between beta1 and beta2
    beta_t = (beta2 - beta1) * torch.arange(0, num_timesteps + 1, dtype=torch.float32) / num_timesteps + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t

    # Compute the log and the cumulative product of the alpha schedule in a numerically stable manner
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # Compute additional terms based on the alpha schedule and its cumulative product
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    # Pack the computed terms into a dictionary and return it
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }
