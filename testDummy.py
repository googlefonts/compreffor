class DummyGlyphSet(dict):
    """Behaves like a glyphset for testing purposes"""

    def __init__(self, *args, **kwargs):
        super(DummyGlyphSet, self).__init__(*args, **kwargs)
        for k in self.keys():
            self[k] = self.DummyCharString(self[k])

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
