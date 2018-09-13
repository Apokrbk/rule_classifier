class Literal:
    def __init__(self, var_name, op, values):
        self.op = op
        self.values = values
        self.var_name = var_name

    def to_string(self):
        string = ""
        string += self.var_name
        string = string + " " + self.op + " "
        string += str(self.values)
        return string

    def value_covered_by_literal(self, value):
        if self.op == '>':
            return value > self.values
        elif self.op == '<':
            return value < self.values
        else:
            return value == self.values or value in self.values

    def count_p_n(self, growset, last_col_name):

        p = 0
        n = 0
        for i in range(0, len(growset[last_col_name])):
            if self.value_covered_by_literal(growset[self.var_name][i]):
                if growset[last_col_name][i] == 1:
                    p += 1
                else:
                    n += 1
        return p, n
