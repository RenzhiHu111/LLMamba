import cv2
import numpy as np
import os


def flowread(flow_path, quantize=False, concat_axis=0, *args, **kwargs):

    if quantize:
        assert concat_axis in [0, 1]
        cat_flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if cat_flow.ndim != 2:
            raise IOError(f'{flow_path} is not a valid quantized flow file, '
                          f'its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)
    else:
        with open(flow_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise IOError(f'Invalid flow file: {flow_path}')
            else:
                if header != 'PIEH':
                    raise IOError(f'Invalid flow file: {flow_path}, '
                                  'header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))

    return flow.astype(np.float32)


def flowwrite(flow, filename, quantize=False, concat_axis=0, *args, **kwargs):

    if not quantize:
        with open(filename, 'wb') as f:
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert concat_axis in [0, 1]
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        os.makedirs(filename, exist_ok=True)
        cv2.imwrite(dxdy, filename)


def quantize_flow(flow, max_val=0.02, norm=True):

    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    # use 255 levels instead of 256 to make sure 0 is 0 after dequantization.
    flow_comps = [
        quantize(d, -max_val, max_val, 255, np.uint8) for d in [dx, dy]
    ]
    return tuple(flow_comps)


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):

    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)

    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def quantize(arr, min_val, max_val, levels, dtype=np.int64):

    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(
            f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    arr = np.clip(arr, min_val, max_val) - min_val
    quantized_arr = np.minimum(
        np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1)

    return quantized_arr


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):

    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(
            f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val -
                                                   min_val) / levels + min_val

    return dequantized_arr
