import numpy as np
import theano.tensor as T
import theano

from .. import init
from .base import Layer
from .input import InputLayer

__all__ = [
    "EmbeddingLayer",
    "MaskedEmbeddingLayer",
    "BinaryInputEmbedding",
    "BatchEmbeddingLayer"
]


class EmbeddingLayer(Layer):
    """
    lasagne.layers.EmbeddingLayer(incoming, input_size, output_size,
    W=lasagne.init.Normal(), **kwargs)

    A layer for word embeddings. The input should be an integer type
    Tensor variable.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings. The last embedding will have index
        input_size - 1.

    output_size : int
        The size of each embedding.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the embedding matrix.
        This should be a matrix with shape ``(input_size, output_size)``.
        See :func:`lasagne.utils.create_param` for more information.

    Examples
    --------
    >>> from lasagne.layers import EmbeddingLayer, InputLayer, get_output
    >>> import theano
    >>> x = T.imatrix()
    >>> l_in = InputLayer((3, ))
    >>> W = np.arange(3*5).reshape((3, 5)).astype('float32')
    >>> l1 = EmbeddingLayer(l_in, input_size=3, output_size=5, W=W)
    >>> output = get_output(l1, x)
    >>> f = theano.function([x], output)
    >>> x_test = np.array([[0, 2], [1, 2]]).astype('int32')
    >>> f(x_test)
    array([[[  0.,   1.,   2.,   3.,   4.],
            [ 10.,  11.,  12.,  13.,  14.]],
    <BLANKLINE>
           [[  5.,   6.,   7.,   8.,   9.],
            [ 10.,  11.,  12.,  13.,  14.]]], dtype=float32)
    """
    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), **kwargs):
        super(EmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]



class BatchEmbeddingLayer(Layer):
    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), mask=None, **kwargs):
        super(BatchEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.mask = mask

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        b, n = input.shape
        out = self.W[input.flatten()].reshape((b, n, self.output_size))
        if self.mask is not None:
            mask = self.mask
            if mask.ndim == 2:
                mask = mask.dimshuffle(0,1,'x').astype(theano.config.floatX)
            out = out * mask
        return out


class MaskedEmbeddingLayer(Layer):
    """
        This is to allow for embeddings which are multihot vectors.

        The general idea:
            1. Index into the rows of the weight matrix 
            2. Use the mask to remove the rows which were padded indices
            3. Sum over the second to last dimension (corresponding to the row indices)
            4. Output the result

        The input could be of size (batch, num_multihot_indices)
        or it could be of size (batch, set_size, num_multihot_indices)
            Here, the set size corresponds to a couple different cases.
            For example, it could be the potential elementary trees which need to be evaluated
    """
    def __init__(self, incoming, mask, input_size, output_size,
                 W=init.Normal(), **kwargs):
        if not isinstance(incoming, Layer):
            assert isinstance(incoming, T.TensorVariable)
            shape = tuple(None for _ in range(incoming.ndim-1)) + (input_size, )
            incoming = InputLayer(shape, incoming)
        super(MaskedEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        # the mask represents the padding needed to make the incoming a uniform size
        self.mask = mask

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        out = self.W[input] * self.mask
        return out.sum(axis=-2)


class BinaryInputEmbedding(Layer):
    def __init__(self, input_shape, input_var, num_units, name=None, W=init.GlorotUniform(), **kwargs):
        incoming = InputLayer(input_shape, input_var, name)
        super(BinaryInputEmbedding, self).__init__(incoming, **kwargs)

        self.W = self.add_param(W, (input_shape[-1], num_units), name="embW")
        self.num_units = num_units



    def get_output_shape_for(self, input_shape):
        return  tuple(self.input_shape[:-1]) + (self.num_units, )

    def get_output_for(self, input, **kwargs):
        if input.ndim  == 3:
            d1, d2, d3 = input.shape
            #assert d3 == self.W.shape[0]
            #assert self.num_units == self.W.shape[1]
            reshaped = input.reshape((d1*d2, d3))
            computed = T.dot(reshaped, self.W)
            out = computed.reshape((d1,d2, self.W.shape[1]))

            #input = T.flatten(input, 2)
        else:
            out = T.dot(input, self.W)

        return out