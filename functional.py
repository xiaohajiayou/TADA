from typing import Any, Union
import torch
from pywt import Wavelet
import torch.nn.functional as F



@torch.jit.script
def _idwt1d(
    lo: torch.Tensor, hi: torch.Tensor, lo_hi: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """performs a 1d idwt on the defined dimension

    Args:
        lo (torch.Tensor): 5d low (average) coefs of shape [N,C,D,H,W]
        hi (torch.Tensor): 5d hi (detail) coefs of shape [N,C,D,H,W]
        lo_hi (torch.Tensor): lo,hi pass filters of shape [2,K]
        dim (int, optional): dimension to apply the idwt to . Defaults to -1.

    Returns:
        torch.Tensor: reconstructed tensor of shape [N,C,D_out,H_out,W_out] (e.g. H_out = 2*H if dim==-1)
    """
    dim = dim % 5

    groups = lo.shape[1]
    filter_c = (
        lo_hi[:, None, None, None, None, :]
        .repeat(1, groups, 1, 1, 1, 1)
        .swapaxes(5, dim + 1)  # swap filter to dwt dim
    )

    # stride of 2 for dwt dim
    stride = [1, 1, 1]
    stride[dim - 2] = 2
    padding = [0, 0, 0]
    padding[dim - 2] = lo_hi.shape[-1] - 2

    a_coefs = F.conv_transpose3d(
        lo, filter_c[0], stride=stride, padding=padding, groups=groups
    )
    d_coefs = F.conv_transpose3d(
        hi, filter_c[1], stride=stride, padding=padding, groups=groups
    )
    return a_coefs + d_coefs


@torch.jit.script
def _dwt1d(x: torch.Tensor, lo_hi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """performs a 1d dwt on the defined dimension

    Args:
        x (torch.Tensor): 4d tensor of shape [N,C,D,H,W]
        lo_hi (torch.Tensor): low and highpass filter (shape [2,K])
        dim (int, optional): dimension to apply the dwt to . Defaults to -1.

    Returns:
        torch.Tensor: dwt coefs of shape [N,2,C,D_out,H_out,W_out]. The average and detail coefs are concatenated in the channels
    """
    dim = dim % 5
    groups = x.shape[1]
    # repeat filter to match number of channels
    filter_c = lo_hi[:, None, None, None, :].repeat(groups, 1, 1, 1, 1).swapaxes(4, dim)

    if x.shape[dim] % 2 != 0:
        # pad dwt dimension to multiple of two
        pad = [0] * 6
        pad[(4 - dim) * 2 + 1] = 1
        x = F.pad(x, pad)

    # stride of 2 for dwt dim
    stride = [1, 1, 1]
    stride[dim - 2] = 2

    padding = [0, 0, 0]
    padding[dim - 2] = lo_hi.shape[-1] - 2

    filtered = F.conv3d(x, filter_c, stride=stride, padding=padding, groups=groups)
    return filtered.reshape(
        filtered.shape[0],
        groups,
        2,
        filtered.shape[2],
        filtered.shape[3],
        filtered.shape[4],
    ).swapaxes(1, 2)



def _to_wavelet_coefs(wavelet):
    # match wavelet:
    #     case str():
    #         return torch.tensor(Wavelet(wavelet).filter_bank)[2:]
    #     case torch.Tensor():
    #         return wavelet
    #     case Wavelet():
    #         return torch.tensor(wavelet.filter_bank)[2:]
    #     case _:
    #         raise Exception("")
    a = Wavelet(wavelet).filter_bank
    return torch.tensor(a)[2:]


def dwt(x: torch.Tensor, wavelet: Union[str,torch.Tensor ,Wavelet] ) -> torch.Tensor:
    """performs the 1D discrete wavelet transform

    Args:
        x (torch.Tensor): input tensor shape [N,C,W]
        wavelet (str | torch.Tensor | Wavelet): wavelet (if tensor [2,C] (lo,hi filter))

    Raises:
        Exception: cannot handle wavelet type

    Returns:
        torch.Tensor: average, detail coefs of shape [N,2,C,W//2]
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    result = _dwt1d(x[:, :, None, None, :], filter, dim=-1)
    return result.reshape(x.shape[0], 2, x.shape[1], -1)


def idwt(x: torch.Tensor, wavelet: Union[str,torch.Tensor ,Wavelet]) -> torch.Tensor:
    """performs the 1D inverse discrete wavelet transform

    Args:
        x (torch.Tensor): [N,2,C,W] average and detail coefs
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: reconstructed tensor
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    result = _idwt1d(
        x[:, 0, :, None, None, :], x[:, 1, :, None, None, :], filter, dim=-1
    )
    return result.squeeze(2).squeeze(2)



class DWT1D(torch.autograd.Function):
    """Performs the 1d dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): input tensor of shape [N,C,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,2,C,W//2]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        x = x.reshape(x.shape[0],x.shape[2],x.shape[1])
        result = _dwt1d(x[:, :, None, None, :], lohi, dim=-1)
        return result.reshape(x.shape[0], 2, x.shape[1], -1)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors
            dx = _idwt1d(
                dx0[:, 0, :, None, None, :], dx0[:, 1, :, None, None, :], lohi
            )[:, :, 0, 0]
        return dx, None


class IDWT1D(torch.autograd.Function):
    """Performs the 1d inverse dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): average and details coefs of shape [N,2,C,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,C,W*2]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        result = _idwt1d(
            x[:, 0, :, None, None, :], x[:, 1, :, None, None, :], lohi, dim=-1
        )
        return result.squeeze(2).squeeze(2)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors

            dx = _dwt1d(dx0[:, :, None, None, :], lohi, -1)[:, :, :, 0, 0]

        return dx, None



if __name__ == '__main__':


    # batch of size 8 with 3 channels
    x = torch.rand(8, 3, 100)
    coefs = dwt(x, "haar")  # coefs of shape (1,2,3,50)
    # reconstruct signal from coefficients
    y = idwt(coefs, "haar")