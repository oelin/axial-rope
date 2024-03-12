def rotate_half(x: torch.Tensor) -> torch.Tensor:

    x1, x2 = x[..., 0::2], x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    x = rearrange(x, '... d r -> ... (d r)')

    return x


def rotate(
    x: torch.Tensor, 
    frequencies: torch.Tensor, 
) -> torch.Tensor:

    truncate = frequencies.size(-1)

    x, x_right = x[..., : truncate], x[..., truncate :]
    x = (x * frequencies.cos()) + (rotate_half(x) * frequencies.sin())
    x = torch.cat((x, x_right), dim=-1)

    return x


class AxialRoPE(nn.Module):
    """Axial RoPE.

    Example
    -------
    >>> module = AxialRoPE(
    ...     embedding_dimension=16,
    ...     heads=16,
    ...     highest_frequency=...,
    ... )
    >>> x = ...  # Shape: (B, H, L, E).
    >>> position = ...  # Shape: (B, L, 2). 
    >>> x = module(x, position)  # Shape: (B, H, L, E).
    """

    def __init__(
        self, 
        *, 
        embedding_dimension: int, 
        heads: int,
        highest_frequency: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension (per head).
        heads : int
            The number of heads.
        highest_frequency : float
            The highest frequency.
        """

        super().__init__()

        frequencies = torch.linspace(
            start=math.log(math.pi),
            end=math.log(highest_frequency * math.pi / 2),
            steps=embedding_dimension // 4,
        ).expand((heads, -1))

        self.x_frequencies = nn.Parameter(frequencies.clone())
        self.y_frequencies = nn.Parameter(frequencies.clone())
    
    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        position : torch.Tensor
            The position tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x_frequencies = position[..., None, None, 0] * self.x_frequencies.exp()
        y_frequencies = position[..., None, None, 1] * self.y_frequencies.exp()

        frequencies = torch.cat((y_frequencies, x_frequencies), dim=-1)
        frequencies = frequencies.repeat_interleave(2, dim=-1)
        frequencies = frequencies.transpose(-2, -3)

        # Rotate.

        x = rotate(x, frequencies)

        return x
