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
            try:
                val = int(self.values)
                return value == self.values
            except (ValueError, TypeError):
                return value in self.values

