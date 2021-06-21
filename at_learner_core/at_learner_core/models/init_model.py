
def get_wrapper(config, wrapper_func=None):
    if wrapper_func is not None:
        wrapper = wrapper_func(config)
        raise Exception('Unknown wrapper architecture type')
    return wrapper
