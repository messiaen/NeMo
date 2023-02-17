# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['CausalConv2D', 'CausalConv1D']


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        self._cache_id = None

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self, x,
    ):
        x = torch.constant_pad_nd(
            x, (self._left_padding, self._right_padding, self._left_padding, self._right_padding), 0
        )
        x = super().forward(x)
        return x


@torch.jit.script
def keep_in_cache_next(cache: torch.Tensor, cache_next: torch.Tensor, cache_keep_size: torch.Tensor, cache_id: int):
    # Current ONNX does not support a Tensor with a dimension of zero
    # Needed to use Torch script to skip this part when this case happens
    if cache_keep_size < cache_next.size(-1):
        # raise Exception("cache_keep_size < cache_next.size(-1)")
        cache_next[cache_id, :, :, :-cache_keep_size] = cache[cache_id, :, :, cache_keep_size:]
    return cache_next


@torch.jit.script
def update_cache_next(
    input_x: torch.Tensor, cache: torch.Tensor, cache_next: torch.Tensor, cache_drop_size: int, cache_id: int
):
    input_x_size = input_x.size(-1) - cache_drop_size
    if input_x_size < 1:
        input_x_size = 1
    input_x_kept = input_x[:, :, :input_x_size]
    cache_keep_size = cache_next.size(-1)
    if cache_keep_size > input_x_size:
        cache_keep_size = input_x_size
    cache_next[cache_id, :, :, :-cache_keep_size] = cache[cache_id, :, :, cache_keep_size:]
    cache_next[cache_id, :, :, -cache_keep_size:] = input_x_kept[:, :, -cache_keep_size:]
    return cache_next


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding
        self._cache_id = None

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache, cache_next):
        # print("cache", cache.size())
        # print("cache_next", cache_next.size())
        # print("x", x.size())
        if cache is None:
            x = torch.constant_pad_nd(x, (self._left_padding, self._right_padding), 0)
        else:
            input_x = x
            needed_cache = cache[self._cache_id, :, :, -self._max_cache_len :]
            # print("needed_cache", needed_cache.size())
            x = torch.constant_pad_nd(x, (0, self._right_padding), 0)
            # print("x post pad", x.size(), F.pad(x, (0, self._right_padding)).size())
            x = torch.cat((needed_cache, x), dim=-1)

            if cache_next is not None:
                cache_next = update_cache_next(input_x, cache, cache_next, self.cache_drop_size, self._cache_id)
        return x

    def forward(self, x, cache, cache_next):
        x = self.update_cache(x=x, cache=cache, cache_next=cache_next)
        x = super().forward(x)
        return x
