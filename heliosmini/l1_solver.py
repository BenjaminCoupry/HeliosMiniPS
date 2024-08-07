import numpy

def weighted_least_squares(A,y,weights):
    """Computes the weighted least squares solution of Ax=y.

    Args:
        A (Array ...,u,v): Design matrix.
        y (Array ...,u): Target values.
        weights (Array ...,u): Weights for each equation.

    Returns:
        Array ...,v : Weighted least squares solution.
    """
    sqrt_weights = numpy.sqrt(weights)
    pinv = numpy.linalg.pinv(A*sqrt_weights[...,numpy.newaxis])
    result = numpy.einsum('...uv,...v->...u',pinv,y*sqrt_weights)
    return result

def iteratively_reweighted_least_squares(A,y, epsilon=1e-5, it=20):
    """Computes the iteratively reweighted least squares solution. of Ax=y

    Args:
        A (Array ..., u, v): Design matrix.
        y (Array ..., u): Target values.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-5.
        it (int, optional): Number of iterations. Defaults to 20.

    Returns:
        Array ..., v: Iteratively reweighted least squares solution.
    """
    weights = numpy.ones(y.shape)
    L2_losses, L1_losses = numpy.zeros(y.shape[:-1]+(it,)), numpy.zeros(y.shape[:-1]+(it,))
    for i in range(it):
        result = weighted_least_squares(A,y,weights)
        ychap = numpy.einsum('...uv,...v->...u',A,result)
        delta = numpy.abs(ychap-y)
        weights = numpy.reciprocal(numpy.maximum(epsilon,delta))
        L2_losses[...,i] = numpy.sum(numpy.square(delta),axis=-1)
        L1_losses[...,i] = numpy.sum(delta,axis=-1)
    return result, L2_losses, L1_losses
