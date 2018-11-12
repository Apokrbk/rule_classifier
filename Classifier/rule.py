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
        return string[:-5]

    def row_covered(self, row):
        if len(self.literals) == 0:
            return True
        covered = True
        for i in range(0, len(self.literals)):
            if not self.literals[i].value_covered_by_literal(row[self.literals[i].var_name]):
                covered = False
                break
        return covered



