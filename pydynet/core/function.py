from .tensor import Tensor, swapaxes


def sqrt(x: Tensor):
    '''平方根函数'''
    return x**0.5


def square(x: Tensor):
    '''平方函数'''
    return x * x


def vsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[0]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[0]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[st:end])

    return sub_tensors


def hsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[1]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[1]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, st:end])

    return sub_tensors


def dsplit(x: Tensor, indices_or_sections: int | tuple) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[2]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[2]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, :, st:end])

    return sub_tensors


def split(
    x: Tensor,
    indices_or_sections: int | tuple,
    axis: int = 0,
) -> list[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    if axis == 0 or axis == -x.ndim:
        return vsplit(x, indices_or_sections)
    elif axis == 1 or axis == -x.ndim + 1:
        return hsplit(x, indices_or_sections)
    elif axis == 2 or axis == -x.ndim + 2:
        return dsplit(x, indices_or_sections)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[axis]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[axis]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = x.xp.array(section_sizes, dtype=x.xp.intp).cumsum()

    sub_tensors = []
    stensor = swapaxes(x, 0, axis)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(swapaxes(stensor[st:end], axis, 0))
    return sub_tensors


def unsqueeze(x: Tensor, axis):
    '''等价于numpy的expand_dims, 因此我们借用了expand_dims的源码'''
    from numpy.core.numeric import normalize_axis_tuple

    if type(axis) not in (tuple, list):
        axis = (axis, )

    out_ndim = len(axis) + x.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(x.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return x.reshape(*shape)


def squeeze(x: Tensor, axis=None):
    shape = x.shape
    if axis is None:
        new_shape = tuple(dim for dim in shape if dim != 1)
    else:
        if isinstance(axis, int):
            axis = (axis, )
        axis = tuple(axis)

        for ax in axis:
            if ax >= len(shape) or ax < -len(shape):
                raise ValueError("Axis out of range")
            if shape[ax] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {ax} with size {shape[ax]}")

        # 构造新形状，排除指定轴
        new_shape = tuple(dim for i, dim in enumerate(shape) if i not in axis)

    # 返回重塑后的数组
    return x.reshape(*new_shape)
