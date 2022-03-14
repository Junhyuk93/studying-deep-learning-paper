import numpy as np

def array_offset(x):
    
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__['data'][0]
    start = x.__array_interface__['data'][0]
    return start - base_start


def calc_pad(pad, in_size, out_size, stride, kernel_size):
    """
    
    Args:
        pad: padding method, "SAME", "VALID", or manually speicified
        kernel_size: kernel size [I, J]
    
    Returns:
        padd_: Actual padding width
    """
    if pad == 'SAME':
        return max((out_size - 1) * stride + kernel_size - in_size, 0)
    elif pad == "VALID":
        return 0
    else:
        return pad

def calc_gradx_pad(pad, in_size, out_size, stride, kernel_size):
    """
    
    Args:
        pad: Padding method, "SAME", "VALID", or manually specified
        in_size: Size of the input to 'conv2d_gradx' (i.e. size of `dy`)
        out_size: Size of the output of 'conv2d_gradx' (i.e. size of `dx`)
        stride: Length of the convolution stride
        kernel_size: Kernel size

    Returns:
        pad_: Actual padding width

    """

    if pad == "SAME":
        out_size_min = (in_size - 1) * stride + 1
        p = out_size + kernel_size - 1 - out_size_min
        p = max(p,0)
        p = min(p, (kernel_size - 1) * 2)
        return p
    elif pad == "VALID":
        return (kernel_size - 1) * 2
    else:
        return pad

def calc_size(h, kh, pad, sh):

    """
    
    Args:
        h: input image size
        kh: kernel size
        pad: padding strategy
        sh: stride

    Returns:
        s: output size

    """

    if pad == "VALID":
        return np.ceil((h - kh + 1) / sh)
    elif pad == "SAME":
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))

def extract_sliding_windows_gradw(x,
                                kernel_size,
                                pad,
                                stride,
                                orig_size,
                                floor_first=True):
    """

    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, H', W', KH, KW, C]

    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]

    kh = kernel_size[0]
    kw = kernel_size[1]
    
    sh = stride[0]
    sw = stride[1]

    h2 = orig_size[0]
    w2 = orig_size[1]

    ph = int(calc_pad(pad, h, h2, 1, ((kh - 1) * sh + 1)))
    pw = int(calc_pad(pad, w, w2, 1, ((kw - 1) * sw + 1)))

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2)) 

    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)

    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)

    x = np.pad(
        x, ((0,0), (ph3, ph2), (pw3, pw2), (0,0)),
        mode = 'constant',
        constant_values=(0.0,))
    p2h = (-x.shape[1]) % sh
    p2w = (-x.shape[2]) % sw

    if p2h > 0 or p2w > 0:
        x = np.pad(
            x, ((0,0), (0, p2h), (0, p2w), (0,0,)),
            mode = 'constant',
            constant_values=(0.0, ))

    # 복사없이 윈도우 추출
    # x = x.reshape([n, int(x.shape[1] / sh), sh, int(x.shape[2] / sw), sw, c])
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         h0 = int(np.floor(ii / sh))
    #         w0 = int(np.floor(jj / sw))
    #         y[:, ii, jj, :, :, :] = x[:, h0:h0 + kh, ii % sh, w0:w0 + kw, jj %
    #                                   sw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, x_sh, x_sw, sh * x_sh, sw * x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                dtype=x.dtype,
                buffer=x.data,
                offset=array_offset(x),
                strides=y_strides)
    return y

def extra