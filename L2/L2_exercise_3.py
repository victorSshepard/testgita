import numpy as np
def linear(w, x):

    # Добавляем столбец единиц к признакам
    x_ext = np.hstack((x, np.ones((x.shape[0], 1))))

    # Вычисляем логиты
    s = np.dot(x_ext, w)


    """
    Calculates logits

    Parameters
    ----------
    w: np.narray
        model weights and bias (parameters) [input_size+1, classes_num]
    x: np.narray
        array of features [batch_size, input_size]

    Returns
    ----------
    x_ext: np.narray
        array of features with ones column [batch_size, input_size+1]
    s: np.narray
        array of logits with size [batch_size, classes_num]
    """
    # Your code here
    return x_ext, s


# tests
# fmt: off
inputs = [
    (
        np.array([[1., 2.,],
                  [3., 4.,],
                  [5., 6.,]]),
        np.array([[1., 2.,],
                  [3., 4.,]])
    ),
    (
        np.array([[1., 1.,],
                  [0., 0.,],
                  [0., 0.,]]),
        np.array([[0., 0.,],
                  [0., 0.,]])
    )
]

outputs = [
    (
        np.array([[1., 1., 2.,],
                  [1., 3., 4.,]]),
        np.array([[14., 18.,],
                  [30., 38.,],]),
    ),
    (
        np.array([[1., 0., 0.,],
                  [1., 0., 0.,]]),
        np.array([[1., 1.,],
                  [1., 1.,]])
    )
]
# fmt: on

for (w, x), (x_out, s_out) in zip(inputs, outputs):
    x, s = linear(w, x)
    assert x.shape == x_out.shape, f"Check shape: {x.shape}!={x_out.shape}"
    assert (np.abs(x - x_out) < 0.00001).all(), f"Check \n{x}\n!=\n{x_out}"
    assert s.shape == s_out.shape, f"Check shape: {s.shape}!={s_out.shape}"
    assert (np.abs(s - s_out) < 0.00001).all(), f"Check \n{s}\n!=\n{s_out}"

print("It's OK")
