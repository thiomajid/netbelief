    print("Max absolute error:", diff.max())
    print("Max relative error:", (diff / (jnp.abs(dummy) + 1e-12)).max())