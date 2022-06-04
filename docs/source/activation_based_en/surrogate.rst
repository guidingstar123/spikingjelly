surrogate gradient
=======================================
Author of this tutorial ： `fangwei123456 <https://github.com/fangwei123456>`_

Translator : `guidingstar123 <https://github.com/guidingstar123>`_

At:doc:`../activation_based/neuron` we've already mentioned that，Describes the process of neuronal firing :math:`S[t] = \Theta(H[t] - V_{threshold})`，A Heaviside step function is used：

.. math::
    \Theta(x) =
    \begin{cases}
    1, & x \geq 0 \\
    0, & x < 0
    \end{cases}

By definition , its derivative is an impulse function：

.. math::
    \delta(x) = 
    \begin{cases}
    +\infty, & x = 0 \\
    0, & x \neq 0
    \end{cases}

Gradient descent using the impulse function directly will obviously make the training of the network and its instability. To solve this problem, various surrogate gradient methods have been proposed，See\
This review `Surrogate Gradient Learning in Spiking Neural Networks <https://arxiv.org/abs/1901.09948>`_。

Alternative functions are used in neurons to generate pulses, which look :class:`BaseNode.neuronal_fire <spikingjelly.activation_based.neuron.BaseNode.neuronal_fire>` source code can be found：

.. code-block:: python

    # spikingjelly.activation_based.neuron
    class BaseNode(base.MemoryModule):
        def __init__(..., surrogate_function: Callable = surrogate.Sigmoid(), ...)
        # ...
        self.surrogate_function = surrogate_function
        # ...
        

        def neuronal_fire(self):
            return self.surrogate_function(self.v - self.v_threshold)


The principle of gradient substitution is to use it in forward propagation :math:`y = \Theta(x)`，Used in backpropagation :math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \sigma'(x)`，Instead\
:math:`\frac{\mathrm{d}y}{\mathrm{d}x} = \Theta'(x)`，thereinto :math:`\sigma(x)` This is the alternative function。:math:`\sigma(x)` Usually a shape with :math:`\Theta(x)` \
Similar, but smooth continuous function.

In :class:`spikingjelly.activation_based.surrogate` Some commonly used alternative functions are provided, including the Sigmoid function :math:`\sigma(x, \alpha) = \frac{1}{1 + \exp(-\alpha x)}` \
is :class:`spikingjelly.activation_based.surrogate.Sigmoid`，The following figure shows the original Heaviside step function ''Heaviside'', 'alpha=5'' for the original Sigmoid proto-function ``Primitive`` \
and its gradient ``Gradient``：

.. image:: ../_static/API/activation_based/surrogate/Sigmoid.*
    :width: 100%


Alternative functions are relatively simple to use, and using alternative functions is like using functions:

.. code-block:: python

    import torch
    from spikingjelly.activation_based import surrogate

    sg = surrogate.Sigmoid(alpha=4.)

    x = torch.rand([8]) - 0.5
    x.requires_grad = True
    y = sg(x)
    y.sum().backward()
    print(f'x={x}')
    print(f'y={y}')
    print(f'x.grad={x.grad}')

The output is：

.. code-block:: shell

    x=tensor([-0.1303,  0.4976,  0.3364,  0.4296,  0.2779,  0.4580,  0.4447,  0.2466],
       requires_grad=True)
    y=tensor([0., 1., 1., 1., 1., 1., 1., 1.], grad_fn=<sigmoidBackward>)
    x.grad=tensor([0.9351, 0.4231, 0.6557, 0.5158, 0.7451, 0.4759, 0.4943, 0.7913])

Each alternative function, in addition to the module-style API of the tangible :class:'spikingjelly.activation_based.surrogate.Sigmoid', also provides a function-style API such as :class:'spikingjelly.activation_based.surrogate.sigmoid'. \
Module-style APIs use hump nomenclature, while function-style APIs use underscore nomenclature, and the relationships are similar ``torch.nn`` and ``torch.nn.functional``，Here are a few examples：

===============  ===============
module              function
===============  ===============
``Sigmoid``      ``sigmoid``
``SoftSign``     ``soft_sign``
``LeakyKReLU``   ``leaky_k_relu``
===============  ===============

The following is an example of the usage of the Function Style API：

.. code-block:: python

    import torch
    from spikingjelly.activation_based import surrogate

    alpha = 4.
    x = torch.rand([8]) - 0.5
    x.requires_grad = True
    y = surrogate.sigmoid.apply(x, alpha)
    y.sum().backward()
    print(f'x={x}')
    print(f'y={y}')
    print(f'x.grad={x.grad}')


Alternative functions will usually have 1 or more hyperparameters that control shapes, for example :``alpha`` of class:`spikingjelly.activation_based.surrogate.Sigmoid` 。\
The shape parameter of the substitution function in SpikingJelly is by default so that the gradient of the substitution function is 1 maximum, which can avoid the gradient explosion problem caused by gradient multiplication to some extent.
