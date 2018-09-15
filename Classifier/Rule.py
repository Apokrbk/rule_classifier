class Rule:
    def __init__(self):
        self.literals = list()

    def add_literal(self, literal):
        self.literals.append(literal)

    def delete_literal(self, literal):
        for i in range(0, len(self.literals)):
            if self.literals[i].to_string() == literal.to_string():
                self.literals.pop(i)
                break

    def to_string(self):
        string = ""
        for literal in self.literals:
            string += literal.to_string()
            string += " and "
        return string[:-4]

    def count_p_n(self, growset):
        p = 0
        n = 0
        last_col_name = growset.columns[len(growset.columns) - 1]
        if len(self.literals) == 0:
            return 0, 0
        for i in range(0, len(growset[last_col_name])):
            covered = True
            for j in range(0, len(self.literals)):
                if not self.literals[j].value_covered_by_literal(growset[self.literals[j].var_name][i]):
                    covered = False
                    break
            if covered:
                if growset[last_col_name][i] == 1:
                    p += 1
                else:
                    n += 1
        return p, n


