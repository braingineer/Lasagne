"""
author: bmcmahan
date: March 4th, 2016
purpose: make layers for use in neural attention experiments
notes: I tried to put the others in files that seemed to capture their behavior
       but these didn't really have an obvious category.. though I guess merge layer
       but i didn't want to put everything into merge.
"""

from .base import MergeLayer
from .. import init, nonlinearities




__all__ = [
    "ContextLayer",
    "SimilarityLayer"
]


class ContextLayer(MergeLayer):
    """ combine attention scores and their vectors
        c_t = sum_j score(a_j) * vec(a_j)
            where score(a_j) = softmax(w*a_j)
        cf.
            Neural Machine Translation by Jointly Learning to Align and Translate
            Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio.
            2014
    """
    def __init__(self, incomings, **kwargs):
        super(ContextLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """ returns (batch,embed_size) """
        att_vec_shape = input_shapes[0]
        return (att_vec_shape[0], att_vec_shape[2])

    def get_output_for(self, inputs, **kwargs):
        """ weight the attention vectors and sum over them; thus summarizing them """
        att_vec, att_score = inputs
        if att_vec.ndim > att_score.ndim:
            assert att_vec.ndim == 3
            assert att_score.ndim == 2
            b,n = att_score.shape
            att_score = att_score.reshape((b,n,1))
        else:
            assert att_score.shape[-1] == 1

        # multiply the scores against the vectors, sum over the options
        # we want a single vector per batch index
        return (att_vec * att_score).sum(axis=1)


class SimilarityLayer(MergeLayer):
    """ assess similarity between multinomial tensor @ conditioning matrix """
    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(SimilarityLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity or nonlinearities.rectify

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1][:2]

    def get_output_for(self, inputs, **kwargs):
        the_matrix,the_tensor = inputs
        b,e = the_matrix.shape
        b,k,e = the_tensor.shape
        the_matrix = the_matrix.reshape((b,1,e))
        similarities = (the_tensor*the_matrix).sum(axis=-1)

        return self.nonlinearity(similarities)

