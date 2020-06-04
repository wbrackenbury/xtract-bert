from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_ext(fpath):

    ext_items = fpath.split(".")
    if len(ext_items) < 2:
        return None

    return ext_items[-1]
