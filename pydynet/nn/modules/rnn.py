from .module import Module
from .. import init
from .. import functional as F
from ..parameter import Parameter
from ...special import empty, zeros
from ... import core
from ...cuda import Device

from typing import Literal, Optional, Tuple, List
import math


class RNNCell(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kwargs = {"device": Device(device), "dtype": dtype}
        self.nonlinearity = nonlinearity
        self.fn = {'tanh': F.tanh, 'relu': F.relu}[nonlinearity]

        self.Wx = Parameter(empty((input_size, hidden_size), **self.kwargs))
        self.Wh = Parameter(empty((hidden_size, hidden_size), **self.kwargs))
        if bias:
            self.bias = Parameter(empty(self.hidden_size, **self.kwargs))
        self.has_bias = bias
        self.reset_paramters()

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x)
        else:
            assert (x.ndim == 1 and h.shape == (self.hidden_size, )) or (
                x.ndim == 2 and h.shape
                == (x.shape[0], self.hidden_size)), "Wrong hidden state input!"

        lin = x @ self.Wx + h @ self.Wh
        if self.has_bias:
            lin = lin + self.bias
        return self.fn(lin)

    def reset_paramters(self):
        bound = math.sqrt(1 / self.hidden_size)
        init.uniform_(self.Wx, -bound, bound)
        init.uniform_(self.Wh, -bound, bound)
        if self.has_bias:
            init.uniform_(self.bias, -bound, bound)

    def init_hidden(self, x):
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            return zeros(self.hidden_size, **self.kwargs)
        else:
            batch_size = x.shape[0]
            return zeros((batch_size, self.hidden_size), **self.kwargs)

    def __repr__(self) -> str:
        return "{}({}, {}, bias={}, nonlinearity={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.has_bias,
            self.nonlinearity,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)


class RNN(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.has_bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.kwargs = {"device": Device(device), "dtype": dtype}

        assert num_layers > 0
        size_list = [input_size] + [hidden_size] * (num_layers - 1)
        self.RNNCells: List[RNNCell] = []
        for i in range(num_layers):
            cell = RNNCell(
                size_list[i],
                hidden_size,
                bias,
                nonlinearity,
                **self.kwargs,
            )
            setattr(self, 'rnn_{}'.format(i), cell)
            self.RNNCells.append(cell)
        if self.bidirectional:
            self.rRNNCells: List[RNNCell] = []
            for i in range(num_layers):
                cell = RNNCell(
                    size_list[i],
                    hidden_size,
                    bias,
                    nonlinearity,
                    **self.kwargs,
                )
                setattr(self, 'rrnn_{}'.format(i), cell)
                self.rRNNCells.append(cell)

    def forward(self, x, h=None):
        if self.batch_first and x.ndim == 3:
            x = x.swapaxes(0, 1)

        if h is None:
            h = self.init_hidden(x)
        else:
            d = 2 if self.bidirectional else 1
            assert (x.ndim == 2
                    and h.shape == (d * self.num_layers, self.hidden_size)
                    ) or (x.ndim == 3 and h.shape
                          == (d * self.num_layers, x.shape[1],
                              self.hidden_size)), "Wrong hidden state input!"

        if self.num_layers == 1 and not self.bidirectional:
            h_list = self.cell_forward(self.RNNCells[0], x, h[0])
            output = core.concat(h_list)
            hn = h_list[-1]

        elif self.num_layers == 1 and self.bidirectional:
            h_list = self.cell_forward(self.RNNCells[0], x, h[0])
            hr_list = self.cell_forward(self.rRNNCells[0], x[::-1], h[1])
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])],
                axis=-1,
            )
            hn = core.concat([h_list[-1], hr_list[-1]])

        elif self.num_layers > 1 and not self.bidirectional:
            hn_list = []
            for i in range(self.num_layers):
                h_list = self.cell_forward(
                    self.RNNCells[i],
                    x if i == 0 else core.concat(h_list),
                    h[i],
                )
                hn_list.append(h_list[-1])
            output = core.concat(h_list)
            hn = core.concat(hn_list)

        else:
            hn_list = []
            hrn_list = []
            for i in range(self.num_layers):
                h_list = self.cell_forward(
                    self.RNNCells[i],
                    x if i == 0 else core.concat(h_list),
                    h[i],
                )
                hr_list = self.cell_forward(
                    self.rRNNCells[i],
                    x[::-1] if i == 0 else core.concat(hr_list),
                    h[i + self.num_layers],
                )
                hn_list.append(h_list[-1])
                hrn_list.append(hr_list[-1])
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])], axis=-1)
            hn = core.concat(hn_list + hrn_list)

        if self.batch_first and x.ndim == 3:
            output = output.swapaxes(0, 1)
            hn = hn.swapaxes(0, 1)
        return output, hn

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.RNNCells[i].reset_paramters()
        if self.bidirectional:
            for i in range(self.num_layers):
                self.rRNNCells[i].reset_paramters()

    def init_hidden(self, x):
        assert x.ndim in {2, 3}
        d = 2 if self.bidirectional else 1
        if x.ndim == 2:
            return zeros(
                (d * self.num_layers, self.hidden_size),
                **self.kwargs,
            )
        else:
            batch_size = x.shape[1]
            return zeros(
                (d * self.num_layers, batch_size, self.hidden_size),
                **self.kwargs,
            )

    def cell_forward(self, cell: RNNCell, x, h):
        seq_len = x.shape[0]
        h_list = []
        for i in range(seq_len):
            h = cell(x[i], h)
            h_list.append(core.unsqueeze(h, axis=0))
        return h_list

    def __repr__(self) -> str:
        return "{}({}, {}, num_layers={}, nonlinearity={}, bias={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.nonlinearity,
            self.has_bias,
            self.batch_first,
            self.bidirectional,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)


class LSTMCell(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kwargs = {"device": Device(device), "dtype": dtype}

        self.Wx = Parameter(empty((input_size, 4 * hidden_size),
                                  **self.kwargs))
        self.Wh = Parameter(
            empty((hidden_size, 4 * hidden_size), **self.kwargs))
        if bias:
            self.bias = Parameter(empty(4 * self.hidden_size, **self.kwargs))
        self.has_bias = bias
        self.reset_paramters()

    def forward(self, x, hx: Optional[Tuple] = None):
        if hx is None:
            h = self.init_hidden(x)
            c = self.init_hidden(x)
        else:
            h, c = hx
            assert (x.ndim == 1 and h.shape == (self.hidden_size, )) or (
                x.ndim == 2 and h.shape
                == (x.shape[0], self.hidden_size)), "Wrong hidden state input!"
            assert (x.ndim == 1 and c.shape == (self.hidden_size, )) or (
                x.ndim == 2 and c.shape
                == (x.shape[0], self.hidden_size)), "Wrong cell state input!"
        lin = x @ self.Wx + h @ self.Wh
        if self.has_bias:
            lin = lin + self.bias
        fio, g = core.hsplit(lin, [3 * self.hidden_size])
        sig_fio, tanh_g = F.sigmoid(fio), F.tanh(g)
        f, i, o = core.hsplit(sig_fio, 3)
        c = f * c + i * tanh_g
        h = o * F.tanh(c)
        return h, c

    def init_hidden(self, x):
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            return zeros(self.hidden_size, **self.kwargs)
        else:
            batch_size = x.shape[0]
            return zeros((batch_size, self.hidden_size), **self.kwargs)

    def reset_paramters(self):
        bound = math.sqrt(1 / self.hidden_size)
        init.uniform_(self.Wx, -bound, bound)
        init.uniform_(self.Wh, -bound, bound)
        if self.has_bias:
            init.uniform_(self.bias, -bound, bound)

    def __repr__(self) -> str:
        return "{}({}, {}, bias={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.has_bias,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)


class LSTM(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.kwargs = {"device": Device(device), "dtype": dtype}

        assert num_layers > 0
        size_list = [input_size] + [hidden_size] * (num_layers - 1)
        self.LSTMCells: List[LSTMCell] = []
        for i in range(num_layers):
            cell = LSTMCell(
                size_list[i],
                hidden_size,
                bias,
                **self.kwargs,
            )
            setattr(self, 'lstm_{}'.format(i), cell)
            self.LSTMCells.append(cell)
        if self.bidirectional:
            self.rLSTMCells: List[LSTMCell] = []
            for i in range(num_layers):
                cell = LSTMCell(
                    size_list[i],
                    hidden_size,
                    bias,
                    **self.kwargs,
                )
                setattr(self, 'rlstm_{}'.format(i), cell)
                self.rLSTMCells.append(cell)

    def forward(self, x, hx: Optional[Tuple] = None):
        if self.batch_first and x.ndim == 3:
            x = x.swapaxes(0, 1)

        if hx is None:
            h = self.init_hidden(x)
            c = self.init_hidden(x)
        else:
            d = 2 if self.bidirectional else 1
            h, c = hx
            assert (x.ndim == 2
                    and h.shape == (d * self.num_layers, self.hidden_size)
                    ) or (x.ndim == 3 and h.shape
                          == (d * self.num_layers, x.shape[1],
                              self.hidden_size)), "Wrong hidden state input!"
            assert (x.ndim == 2
                    and c.shape == (d * self.num_layers, self.hidden_size)
                    ) or (x.ndim == 3 and c.shape
                          == (d * self.num_layers, x.shape[1],
                              self.hidden_size)), "Wrong cell state input!"

        if self.num_layers == 1 and not self.bidirectional:
            h_list, c_list = self.cell_forward(
                self.LSTMCells[0],
                x,
                (h[0], c[0]),
            )
            output = core.concat(h_list)
            hn = h_list[-1]
            cn = c_list[-1]
        elif self.num_layers == 1 and self.bidirectional:
            h_list, c_list = self.cell_forward(
                self.LSTMCells[0],
                x,
                (h[0], c[0]),
            )
            hr_list, cr_list = self.cell_forward(
                self.rLSTMCells[0],
                x[::-1],
                (h[1], c[1]),
            )
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])], axis=-1)
            hn = core.concat([h_list[-1], hr_list[-1]])
            cn = core.concat([c_list[-1], cr_list[-1]])
        elif self.num_layers > 1 and not self.bidirectional:
            hn_list, cn_list = [], []
            for i in range(self.num_layers):
                h_list, c_list = self.cell_forward(
                    self.LSTMCells[i],
                    x if i == 0 else core.concat(h_list),
                    (h[i], c[i]),
                )
                hn_list.append(h_list[-1])
                cn_list.append(c_list[-1])
            output = core.concat(h_list)
            hn = core.concat(hn_list)
            cn = core.concat(cn_list)
        else:
            hn_list, hrn_list = [], []
            cn_list, crn_list = [], []
            for i in range(self.num_layers):
                h_list, c_list = self.cell_forward(
                    self.LSTMCells[i],
                    x if i == 0 else core.concat(h_list),
                    (h[i], c[i]),
                )
                hr_list, cr_list = self.cell_forward(
                    self.rLSTMCells[i],
                    x[::-1] if i == 0 else core.concat(hr_list),
                    (h[i + self.num_layers], c[i + self.num_layers]),
                )
                hn_list.append(h_list[-1])
                hrn_list.append(hr_list[-1])
                cn_list.append(c_list[-1])
                crn_list.append(cr_list[-1])
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])], axis=-1)
            hn = core.concat(hn_list + hrn_list)
            cn = core.concat(cn_list + crn_list)
        if self.batch_first and x.ndim == 3:
            output = output.swapaxes(0, 1)
            hn = hn.swapaxes(0, 1)
            cn = cn.swapaxes(0, 1)

        return output, (hn, cn)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.LSTMCells[i].reset_paramters()
        if self.bidirectional:
            for i in range(self.num_layers):
                self.rLSTMCells[i].reset_paramters()

    def init_hidden(self, x):
        assert x.ndim in {2, 3}
        d = 2 if self.bidirectional else 1
        if x.ndim == 2:
            return zeros(
                (d * self.num_layers, self.hidden_size),
                **self.kwargs,
            )
        else:
            batch_size = x.shape[1]
            return zeros(
                (d * self.num_layers, batch_size, self.hidden_size),
                **self.kwargs,
            )

    def cell_forward(self, cell: RNNCell, x, h: Tuple):
        seq_len = x.shape[0]
        h_list, c_list = [], []
        for i in range(seq_len):
            h = cell(x[i], h)  # Infact, `h` here is a tuple (h, c)
            h_list.append(core.unsqueeze(h[0], axis=0))
            c_list.append(core.unsqueeze(h[1], axis=0))
        return h_list, c_list

    def __repr__(self) -> str:
        return "{}({}, {}, num_layers={}, bias={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.has_bias,
            self.batch_first,
            self.bidirectional,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)


class GRUCell(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kwargs = {"device": Device(device), "dtype": dtype}

        self.Wx1 = Parameter(
            empty((input_size, 2 * hidden_size), **self.kwargs))
        self.Wh1 = Parameter(
            empty((hidden_size, 2 * hidden_size), **self.kwargs))
        self.Wx2 = Parameter(empty((input_size, hidden_size), **self.kwargs))
        self.Wh2 = Parameter(empty((hidden_size, hidden_size), **self.kwargs))

        if bias:
            self.bias1 = Parameter(empty(2 * self.hidden_size, **self.kwargs))
            self.bias2 = Parameter(empty(self.hidden_size, **self.kwargs))

        self.has_bias = bias
        self.reset_parameters()

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x)
        else:
            assert (x.ndim == 1 and h.shape == (self.hidden_size, )) or (
                x.ndim == 2 and h.shape
                == (x.shape[0], self.hidden_size)), "Wrong hidden state input!"

        lin1 = x @ self.Wx1 + h @ self.Wh1
        if self.has_bias:
            lin1 = lin1 + self.bias1
        z, r = core.split(F.sigmoid(lin1), 2, axis=1)
        lin2 = x @ self.Wx2 + (r * h) @ self.Wh2
        if self.has_bias:
            lin2 = lin2 + self.bias2
        return (1 - z) * h + z * F.tanh(lin2)

    def reset_parameters(self):
        bound = math.sqrt(1 / self.hidden_size)
        init.uniform_(self.Wx1, -bound, bound)
        init.uniform_(self.Wx2, -bound, bound)
        init.uniform_(self.Wh1, -bound, bound)
        init.uniform_(self.Wh2, -bound, bound)
        if self.has_bias:
            init.uniform_(self.bias1, -bound, bound)
            init.uniform_(self.bias2, -bound, bound)

    def init_hidden(self, x):
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            return zeros(self.hidden_size, **self.kwargs)
        else:
            batch_size = x.shape[0]
            return zeros((batch_size, self.hidden_size), **self.kwargs)

    def __repr__(self) -> str:
        return "{}({}, {}, bias={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.has_bias,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)


class GRU(Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.kwargs = {"device": Device(device), "dtype": dtype}

        assert num_layers > 0
        size_list = [input_size] + [hidden_size] * (num_layers - 1)
        self.GRUCells: List[GRUCell] = []
        for i in range(num_layers):
            cell = GRUCell(
                size_list[i],
                hidden_size,
                bias,
                **self.kwargs,
            )
            setattr(self, 'gru_{}'.format(i), cell)
            self.GRUCells.append(cell)
        if self.bidirectional:
            self.rGRUCells: List[GRUCell] = []
            for i in range(num_layers):
                cell = GRUCell(
                    size_list[i],
                    hidden_size,
                    bias,
                    **self.kwargs,
                )
                setattr(self, 'rgru_{}'.format(i), cell)
                self.rGRUCells.append(cell)

    def forward(self, x, h=None):
        if self.batch_first and x.ndim == 3:
            x = x.swapaxes(0, 1)

        if h is None:
            h = self.init_hidden(x)
        else:
            d = 2 if self.bidirectional else 1
            assert (x.ndim == 2
                    and h.shape == (d * self.num_layers, self.hidden_size)
                    ) or (x.ndim == 3 and h.shape
                          == (d * self.num_layers, x.shape[1],
                              self.hidden_size)), "Wrong hidden state input!"

        if self.num_layers == 1 and not self.bidirectional:
            h_list = self.cell_forward(self.GRUCells[0], x, h[0])
            output = core.concat(h_list)
            hn = h_list[-1]

        elif self.num_layers == 1 and self.bidirectional:
            h_list = self.cell_forward(self.GRUCells[0], x, h[0])
            hr_list = self.cell_forward(self.rGRUCells[0], x[::-1], h[1])
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])], axis=-1)
            hn = core.concat([h_list[-1], hr_list[-1]])

        elif self.num_layers > 1 and not self.bidirectional:
            hn_list = []
            for i in range(self.num_layers):
                h_list = self.cell_forward(
                    self.GRUCells[i],
                    x if i == 0 else core.concat(h_list),
                    h[i],
                )
                hn_list.append(h_list[-1])
            output = core.concat(h_list)
            hn = core.concat(hn_list)

        else:
            hn_list = []
            hrn_list = []
            for i in range(self.num_layers):
                h_list = self.cell_forward(
                    self.GRUCells[i],
                    x if i == 0 else core.concat(h_list),
                    h[i],
                )
                hr_list = self.cell_forward(
                    self.rGRUCells[i],
                    x[::-1] if i == 0 else core.concat(hr_list),
                    h[i + self.num_layers],
                )
                hn_list.append(h_list[-1])
                hrn_list.append(hr_list[-1])
            output = core.concat(
                [core.concat(h_list),
                 core.concat(hr_list[::-1])], axis=-1)
            hn = core.concat(hn_list + hrn_list)

        if self.batch_first and x.ndim == 3:
            output = output.swapaxes(0, 1)
            hn = hn.swapaxes(0, 1)
        return output, hn

    def init_hidden(self, x):
        assert x.ndim in {2, 3}
        d = 2 if self.bidirectional else 1
        if x.ndim == 2:
            return zeros(
                (d * self.num_layers, self.hidden_size),
                **self.kwargs,
            )
        else:
            return zeros(
                (d * self.num_layers, x.shape[1], self.hidden_size),
                **self.kwargs,
            )

    def cell_forward(self, cell: GRUCell, x, h):
        seq_len = x.shape[0]
        h_list = []
        for i in range(seq_len):
            h = cell(x[i], h)
            h_list.append(core.unsqueeze(h, axis=0))
        return h_list

    def __repr__(self) -> str:
        return "{}({}, {}, num_layers={}, bias={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.has_bias,
            self.batch_first,
            self.bidirectional,
        )

    def move(self, device):
        self.kwargs['device'] = device
        return super().move(device)
