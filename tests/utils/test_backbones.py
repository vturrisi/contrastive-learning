# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)


def test_backbones():
    # swin models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = swin_tiny(window_size=4, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_small()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_base()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_large()
    assert isinstance(model(dummy_data), torch.Tensor)

    # vit models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = vit_tiny(patch_size=8, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_small()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_base()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_large()
    assert isinstance(model(dummy_data), torch.Tensor)
