import tensorflow as tf
import numpy as np
from tensorflow_probability.python.math.interpolation import interp_regular_1d_grid

class ANNModel(object):
    """
    wrapper around model.predict() to apply X and Y scalers
    """
    def __init__(self):
        pass

    def load_model(self, file):
        self.model = tf.keras.models.load_model(file)

    def load_X_scalers(self, filename, allow_pickle=True):
        self.X_scalers = np.load(filename, allow_pickle=allow_pickle)

    def load_Y_scalers(self, filename, allow_pickle=True):
        self.Y_scalers = np.load(filename, allow_pickle=allow_pickle)

    def predict(self, X):
        if self.scaleX:
            X = scale.apply_scaler(X, self.X_scalers)

        y = self.model.predict(X)

        if self.scaleY:
            y = scale.apply_inverse_scaler(y, self.Y_scalers)

        return y

def load_model(basis_file, nn_weights_file, X_scalers_file, Y_scalers_file):
    model = ANNModel()
    model.load_model(nn_weights_file)

    if X_scalers_file:
        model.scaleX = True
        model.load_X_scalers(X_scalers_file)
    else:
        model.scaleX = False

    if Y_scalers_file:
        model.scaleY = True
        model.load_Y_scalers(Y_scalers_file)
    else:
        model.scaleY = False

    basis = np.load(basis_file)

    return model, basis

@tf.function(experimental_compile=True, autograph=True, experimental_relax_shapes=True)
def predict_hack(model, _input, scaleX=True, scaleY=True):
    """
    Function to re-create model.predict(input) but allows the graph to be built correctly
    (see https://github.com/tensorflow/tensorflow/issues/33997)

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to make a prediction
    :return:
    """
    x = _input
    if scaleX:
        x = x_scale_hack(model, x)
    pred = tf.convert_to_tensor(x)
    # the three should be an argument for ndim of problem
    pred = tf.reshape(pred, shape=(-1, 3))
    layers = model.model.layers
    for layer in layers:
        pred = layer(pred)
    pred = tf.reshape(
        pred, shape=(-1, model.model.get_output_shape_at(-1)[1],))
    if scaleY:
        #pred = y_inv_scale_hack(model, pred)
        pred = y_inv_minmax_scale_hack(model, pred)
    return pred


# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def x_scale_hack(model, _input):
    """
    Function to re-create sklearn standard scaler for the input (as a tensorflow function)

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    x = _input
    means = tf.constant([model.X_scalers[i].mean_[0]
                         for i in range(x.shape[0])])
    stds = tf.constant([model.X_scalers[i].scale_[0]
                        for i in range(x.shape[0])])
    means = tf.cast(means, tf.float32)
    stds = tf.cast(stds, tf.float32)

    x_scaled = (tf.reshape(x, shape=(-1, x.shape[0],)) - means) / stds
    x_scaled = tf.cast(x_scaled, tf.float32)
    return x_scaled


# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def get_model_y_std_scalers(model, _input):
    """
    Re-casts the sklearn output std scalers for each output basis

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    std = [model.Y_scalers[i].scale_[0] for i in range(_input.shape[1])]
    std = tf.stack(std, axis=0)
    std = tf.cast(std, tf.float32)
    return std


# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def get_model_y_mean_scalers(model, _input):
    """
    Re-casts the sklearn output mean scalers for each output basis

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    mean = [model.Y_scalers[i].mean_[0] for i in range(_input.shape[1])]
    mean = tf.stack(mean, axis=0)
    mean = tf.cast(mean, tf.float32)
    return mean


# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def y_inv_scale_hack(model, _input):
    """
    Re-creates the sklearn inverse standard scaler as a tensorflow function,

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    y_scaled = _input
    y_scaled = tf.reshape(y_scaled, shape=(-1, y_scaled.shape[1]))

    mean = get_model_y_mean_scalers(model, _input)

    std = get_model_y_std_scalers(model, _input)
    y = y_scaled * std + mean
    return y



# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def get_model_y_minmax_scale_scalers(model, _input):
    """
    Re-casts the sklearn output scale scalers for each output basis

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    scale = [model.Y_scalers[i].scale_[0] for i in range(_input.shape[1])]
    scale = tf.stack(scale, axis=0)
    scale = tf.cast(scale, tf.float32)
    return scale


# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def get_model_y_minmax_min_scalers(model, _input):
    """
    Re-casts the sklearn output min scalers for each output basis

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    mins = [model.Y_scalers[i].min_[0] for i in range(_input.shape[1])]
    mins = tf.stack(mins, axis=0)
    mins = tf.cast(mins, tf.float32)
    return mins



# @tf.function(experimental_compile=True, experimental_relax_shapes=True)
def y_inv_minmax_scale_hack(model, _input):
    """
    Re-creates the sklearn inverse standard scaler as a tensorflow function,

    :param model: A trained scrinet Neural Network
    :param _input: The point at which you want to scale
    :return:
    """
    y_scaled = _input
    y_scaled = tf.reshape(y_scaled, shape=(-1, y_scaled.shape[1]))

    mins= get_model_y_minmax_min_scalers(model, _input)

    scale = get_model_y_minmax_scale_scalers(model, _input)
    y = (y_scaled - mins) / scale
    return y


@tf.function(experimental_compile=True, experimental_relax_shapes=True)
def generate_surrogate(x,
                       amp_model,
                       amp_basis,
                       phase_model,
                       phase_basis,
                       amp_scaleX=True, amp_scaleY=False, phase_scaleX=True, phase_scaleY=True):
    """
    Predict a full surrogate waveform for a given position and basis

    :param x: The location in parameter space - currently (q, chi1, chi2)
    :param amp_model: The pre-loaded amplitude nn generator
    :param amp_basis: The pre-loaded amplitude basis
    :param phase_model: The pre-loaded phase nn generator
    :param phase_basis: The pre-loaded phase basis
    :return:
    """

    x = tf.transpose(
        tf.stack([tf.math.log(x[:, 0]), x[:, 1], x[:, 2]], axis=0))
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, shape=(-1, x.shape[0]))
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    amp_alpha = predict_hack(amp_model, x, scaleX=amp_scaleX, scaleY=amp_scaleY)

    amp = tf.tensordot(amp_alpha,
                       amp_basis,
                       axes=1)

    phase_alpha = predict_hack(phase_model,
                               x, scaleX=phase_scaleX, scaleY=phase_scaleY)

    phase = tf.tensordot(phase_alpha,
                         phase_basis,
                         axes=1)

    phase = tf.cast(phase, tf.complex64)
    amp = tf.cast(amp, tf.complex64)

    h = amp * tf.math.exp(-1.j * phase)
    phase = tf.cast(phase, tf.float32)
    phase = phase[0]
    phase = phase - phase[0]

    return tf.math.real(h), tf.math.imag(h), amp, phase

@tf.function(experimental_compile=True, experimental_relax_shapes=True)
def gen_sur_and_interpolate(x,
                       amp_model,
                       amp_basis,
                       phase_model,
                       phase_basis):
    _, _, amp, phase = generate_surrogate(x, amp_model, amp_basis, phase_model, phase_basis)
    amp = tf.cast(amp, tf.float32)

    # num = 33205 = 20100*1.652 # for mtot=60 and deltaT=1./2048
    times = tf.linspace(start=-20000., stop=100., num=33205)

    new_amp = interp_regular_1d_grid(x=times, x_ref_min=-20000., x_ref_max=100., y_ref=amp)
    new_phase = interp_regular_1d_grid(x=times, x_ref_min=-20000., x_ref_max=100., y_ref=phase)

    return new_amp, new_phase

def time_of_frequency():
    """
    estimate time of frequency using phenomD
    used to get start time for surrogate
    """
    pass


@tf.function(experimental_compile=True, experimental_relax_shapes=True)
def generate_surrogate_new(x,
                       amp_model,
                       amp_basis,
                       phase_model,
                       phase_basis,
                       amp_scaleX=True,
                       amp_scaleY=False,
                       phase_scaleX=True,
                       phase_scaleY=True,
                       time_array=None):
    """
    time_array: tf tensor of times. If None then use default times
    otherwise will interpolate amp and phase onto this array.
    """
    pass
