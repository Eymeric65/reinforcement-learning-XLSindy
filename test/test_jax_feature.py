"""
just want to check some stuff concerning jax 
"""

import jax.numpy as jnp
from jax import jit
from jax import vmap
from jax import lax


def test_bool_1(t):

    if t>10:

        return t 
    else:
        return t*2
    
def test_bool_2(t):

    return t*2
    
def test_bool_3(t):
    return lax.cond(t > 10,  # Condition
                    lambda t: t,  # True branch
                    lambda t: t * 2,  # False branch
                    t)  # Operand passed to the branches
    
test_bool_v = vmap(test_bool_3,in_axes=(0),out_axes=(0))

test_bool_v = jit(test_bool_v)

print(test_bool_v(jnp.array([2,3,5,13,23,3])))
