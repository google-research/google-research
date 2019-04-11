import re as RE

class ArgParser(object):
    global_parser = None

    def __init__(self):
        self._table = dict()
        return

    def clear(self):
        self._table.clear()
        return

    def load_args(self, arg_strs):
        succ = True
        vals = []
        curr_key = ''

        for str in arg_strs:
            if not (self._is_comment(str)):
                is_key = self._is_key(str)
                if (is_key):
                    if (curr_key != ''):
                        if (curr_key not in self._table):
                            self._table[curr_key] = vals

                    vals = []
                    curr_key = str[2::]
                else:
                    vals.append(str)

        if (curr_key != ''):
            if (curr_key not in self._table):
                self._table[curr_key] = vals

            vals = []

        return succ

    def load_file(self, filename):
        succ = False
        with open(filename, 'r') as file:
            lines = RE.split(r'[\n\r]+', file.read())
            file.close()

            arg_strs = []
            for line in lines:
                if (len(line) > 0 and not self._is_comment(line)):
                    arg_strs += line.split()

            succ = self.load_args(arg_strs)
        return succ

    def has_key(self, key):
        return key in self._table

    def parse_string(self, key, default=''):
        str = default
        if self.has_key(key):
            str = self._table[key][0]
        return str

    def parse_strings(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = self._table[key]
        return arr

    def parse_int(self, key, default=0):
        val = default
        if self.has_key(key):
            val = int(self._table[key][0])
        return val

    def parse_ints(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [int(str) for str in self._table[key]]
        return arr

    def parse_float(self, key, default=0.0):
        val = default
        if self.has_key(key):
            val = float(self._table[key][0])
        return val

    def parse_floats(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [float(str) for str in self._table[key]]
        return arr

    def parse_bool(self, key, default=False):
        val = default
        if self.has_key(key):
            val = self._parse_bool(self._table[key][0])
        return val

    def parse_bools(self, key, default=[]):
        arr = default
        if self.has_key(key):
            arr = [self._parse_bool(str) for str in self._table[key]]
        return arr

    def _is_comment(self, str):
        is_comment = False
        if (len(str) > 0):
            is_comment = str[0] == '#'

        return is_comment
        
    def _is_key(self, str):
        is_key = False
        if (len(str) >= 3):
            is_key = str[0] == '-' and str[1] == '-'

        return is_key

    def _parse_bool(self, str):
        val = False
        if (str == 'true' or str == 'True' or str == '1' 
            or str == 'T' or str == 't'):
            val = True
        return val