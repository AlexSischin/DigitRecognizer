import numpy as np


class DomainError(Exception):
    pass


class Function:
    def get_domain(self) -> tuple[float, float]:
        raise NotImplemented()

    def at(self, x) -> float:
        raise NotImplemented()

    def at_vec(self, x_vec):
        raise NotImplemented()


class FormulaFunction(Function):
    def __init__(self, func, domain=(-np.inf, +np.inf)) -> None:
        self._domain = domain
        self.at = func
        self.at_vec = np.vectorize(func, otypes=[float])

    def get_domain(self):
        return self._domain[0], self._domain[1]


class ApproximateFunction(Function):
    def __init__(self, x, y):
        if x.ndim != 1 or x.size < 2 or x.shape != y.shape:
            raise ValueError(f'X and Y must be 1D arrays of the same length > 1. Got shapes: {x.shape, y.shape}')
        if not np.all(x[:-1] < x[1:]):
            raise ValueError('X must be strictly increasing')
        self._x = x
        self._y = y
        self.at_vec = np.vectorize(self.at, otypes=[float])

    def get_domain(self):
        return self._x[0], self._x[-1]

    def at(self, x):
        i_r = np.searchsorted(self._x, x, side='right')
        i_l = i_r - 1
        try:
            x_l, x_r = self._x[i_l], self._x[i_r]
            y_l, y_r = self._y[i_l], self._y[i_r]
        except IndexError:
            if x == self._x[-1]:
                return self._y[-1]
            else:
                raise DomainError(f'Domain does not include {x}')
        return y_l + (x - x_l) * (y_r - y_l) / (x_r - x_l)


def transform_distribution(domain: np.ndarray, x_pdf: Function, t_inv: Function,
                           t_der: Function) -> ApproximateFunction:
    der_lb, der_rb = t_der.get_domain()
    x_pdf_lb, x_pdf_rb = x_pdf.get_domain()
    t_inv_lb, t_inv_rb = t_inv.get_domain()

    a = np.searchsorted(domain, t_inv_lb, side='right') if t_inv_lb > domain[0] else 0
    b = np.searchsorted(domain, t_inv_rb, side='left') if t_inv_rb < domain[-1] else -1

    sub_x = domain[a:b]
    inv = t_inv.at_vec(sub_x)

    x_pdf_args_i = (inv >= x_pdf_lb) & (inv <= x_pdf_rb)
    x_pdf_args = inv[x_pdf_args_i]
    x_pdf_values = np.zeros(sub_x.shape)
    x_pdf_values[x_pdf_args_i] = x_pdf.at_vec(x_pdf_args)

    der_args_i = (inv >= der_lb) & (inv <= der_rb)
    der_args = inv[der_args_i]
    der_values = np.zeros(sub_x.shape)
    der_values[der_args_i] = t_der.at_vec(der_args)

    sub_y = x_pdf_values / der_values

    y = np.zeros(domain.shape, dtype=float)
    y[a:b] = sub_y
    return ApproximateFunction(domain, y)


def sum_distribution_np(domain: np.ndarray, x_pdf: Function, y_pdf: Function) -> ApproximateFunction:
    fx = x_pdf.at_vec(domain)
    fy = y_pdf.at_vec(domain)
    ft = np.convolve(fx, fy, 'same')

    dx = domain[1] - domain[0]
    area = np.trapz(ft, dx=dx)
    ft = ft / area

    return ApproximateFunction(domain, ft)


def sum_distribution(domain: np.ndarray, x_pdf: Function, y_pdf: Function) -> ApproximateFunction:
    x_i = domain
    dx = x_i[1] - x_i[0]
    n = x_i.size - 1
    r_r = int(np.floor(x_i[n] / dx))
    r_l = int(np.ceil(x_i[0] / dx))
    k = np.arange(0, r_r - r_l + 1)
    x_k = (k + r_l) * dx

    fx_x_i = x_pdf.at_vec(x_i)
    fy_x_k = y_pdf.at_vec(x_k)

    ft_t_j = np.zeros(shape=(n + 1))
    for j in range(ft_t_j.size):
        i_a = max(0, j - r_r)
        i_b = min(n, j - r_l)
        k_a = j - i_a - r_l
        k_b = j - i_b - r_l
        fx_x_i_sub = fx_x_i[i_a: i_b + 1]
        fy_x_k_sub = np.flip(fy_x_k[k_b: k_a + 1])
        prods = fx_x_i_sub * fy_x_k_sub
        y = np.trapz(prods, dx=dx)
        ft_t_j[j] = y

    area = np.trapz(ft_t_j, dx=dx)
    ft_t_j = ft_t_j / area

    return ApproximateFunction(domain, ft_t_j)


def product_distribution(domain: np.ndarray, x_pdf: Function, y_pdf: Function) -> ApproximateFunction:
    x_i = domain
    dx = x_i[1] - x_i[0]
    n = x_i.size - 1
    x_nonzero = x_i != 0

    fx_x_i = x_pdf.at_vec(x_i)
    abs_x_i = np.divide(1, np.abs(x_i), where=x_nonzero)
    ft_t_j = np.zeros(shape=(n + 1))
    for j in range(ft_t_j.size):
        all_t_div_x = np.divide(x_i[j], x_i, where=x_nonzero)
        i_valid = (~np.isnan(all_t_div_x)) & (all_t_div_x >= x_i[0]) & (all_t_div_x <= x_i[n])
        if i_valid.size != 0:
            f_y_t_div_x = y_pdf.at_vec(all_t_div_x[i_valid])
            prods = fx_x_i[i_valid] * f_y_t_div_x * abs_x_i[i_valid]
            y = np.trapz(prods, dx=dx)
        else:
            y = 0
        ft_t_j[j] = y

    area = np.trapz(ft_t_j, dx=dx)
    ft_t_j = ft_t_j / area

    return ApproximateFunction(domain, ft_t_j)


def sigmoid_inv(y):
    return np.log(y / (1 - y))


def sigmoid_der(x):
    sigmoid_of_x = 1 / (1 + np.e ** (-x))
    return sigmoid_of_x * (1 - sigmoid_of_x)


def get_relu_inv_approx_func(k):
    def relu_inv_approx(y):
        np.log(np.e ** (k * y) - 1) / k

    return relu_inv_approx


def get_relu_der_approx_func(k):
    def relu_der_approx(x):
        return 1 / (1 + np.e ** (-x * k))

    return relu_der_approx


def get_uniform_pdf(a, b):
    v = 1 / (b - a)

    def uniform(x):
        return v if a <= x <= b else 0.0

    return uniform


def get_gaussian_pdf(m, sd):
    k = 1 / (sd * np.sqrt(2 * np.pi))
    r = -1 / (2 * sd * sd)

    def normal(x):
        return k * np.exp(r * (x - m) ** 2)

    return normal
