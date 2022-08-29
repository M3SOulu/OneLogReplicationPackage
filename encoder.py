import string


class CharacterEncoder:
    def __init__(self):
        self.encodings = {char: index for index, char in
                          enumerate(string.ascii_letters + string.digits + string.punctuation, start=1)}

        self.decoding = {v: k for k, v in self.encodings.items()}

    def encode(self, x):
        if type(x) is str:
            return [self[c] for c in x]
        return [self.encode(e) for e in x]

    def decode(self, x):
        if type(x) is list and type(x[0]) is int:
            return ''.join(self[c] for c in x)
        return [self.decode(e) for e in x]

    def all_keys(self):
        return ["<PAD>"] + list(self.encodings.keys()) + ["<UCT>"]

    @property
    def dictionary_length(self):
        return len(self.encodings) + 2

    def __getitem__(self, item):
        if item == 0:
            return 0
        if item in self.encodings:
            return self.encodings[item]
        if item in self.decoding:
            return self.decoding[item]
        if type(item) is int:
            return ' '
        if type(item) is str:
            return len(self.encodings) + 1
