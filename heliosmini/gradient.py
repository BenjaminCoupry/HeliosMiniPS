import optax
import jax
import functools


@functools.partial(
    jax.jit, static_argnames=["optimizer", "loss", "feeder", "iterations", "callback"]
)
def gradient_descent(
    optimizer,
    loss,
    feeder,
    parameters,
    feeder_state,
    iterations,
    unroll=10,
    callback = None,
    **kwargs
):
    """
    Perform gradient descent optimization using JAX and Optax.

    Args:
        optimizer: An Optax optimizer instance used for updating parameters.
        loss: A function that computes the value of the objective function.
        parameters: Initial parameters for the optimization.
        refreshed_parameters: Additional parameters that might be refreshed or updated during optimization.
        iterations: Number of iterations to run the gradient descent.
        unroll: Number of iterations to unroll in the loop (default is 100).
        **kwargs: Additional keyword arguments to pass to `value_and_grad`.

    Returns:
        A tuple containing:
        - Updated parameters after optimization.
        - Refreshed parameters after optimization.
        - Losses recorded during each iteration.
    """
    # Create a partial function for computing value and gradient with additional arguments.
    partial_gradient = functools.partial(jax.value_and_grad(loss), **kwargs)

    def body_fun(i, val):
        """
        The body function for the JAX fori_loop.

        Args:
            i: Current iteration index.
            val: A tuple containing current optimizer state, parameters, refreshed parameters, and losses.

        Returns:
            A tuple with updated optimizer state, parameters, refreshed parameters, and losses.
        """
        
        opt_state, parameters, feeder_state, losses = val
        #compute the gradient args with the feeder
        gradient_args, feeder_state = feeder(parameters, feeder_state)
        # Compute the value and gradient of the objective function.
        value, grad = partial_gradient(
            parameters, **gradient_args
        )
        # Update the losses array with the current value.
        losses = losses.at[i].set(value)
        # Compute the parameter updates and update the optimizer state.
        updates, opt_state = optimizer.update(grad, opt_state)
        # Apply the updates to the parameters.
        parameters = optax.apply_updates(parameters, updates)
        if callback is not None:
            jax.lax.cond(jax.numpy.mod(i,100) == 0, lambda i, value : jax.debug.callback(callback, i, value), lambda i, value : None, i, value)
        return (opt_state, parameters, feeder_state, losses)

    # Initialize the optimizer state.
    opt_state = optimizer.init(parameters)
    # Initialize the losses array to store the loss value at each iteration.
    losses = jax.numpy.zeros((iterations,))

    # Run the optimization loop for the specified number of iterations.
    _, parameters, _, losses = jax.lax.fori_loop(
        0,
        iterations,
        body_fun,
        (opt_state, parameters, feeder_state, losses),
        unroll=unroll,
    )

    return parameters, losses
