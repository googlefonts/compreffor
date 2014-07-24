import collections

class DummyGlyphSet(collections.MutableMapping):
    """Behaves like a glyphset for testing purposes"""

    def __init__(self, *args, **kwargs):
        self.storage = {}
        self.update(dict(*args, **kwargs)) # interpret initial args

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, value):
        self.storage[key] = self.DummyCharString(value)

    def __delitem__(self, key):
        del self.storage[key]

    def __iter__(self):
        return iter(self.storage)

    def __len__(self):
        return len(self.storage)

    class DummyCharString(object):
        program = None

        def __init__(self, data):
            self.program = data

        def decompile(self):
            pass

        def __iter__(self):
            return iter(self.program)

        def __repr__(self):
            return repr(self.program)

        def __str__(self):
            return str(self.program)

        def __len__(self):
            return len(self.program)
