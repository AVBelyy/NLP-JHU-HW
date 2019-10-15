input_file = "atis-grammar-cnf.cfg"
output_file = "atis.gr"

rules = {}
with open(input_file) as f:
    for line in f.readlines():
        lhs, rhs = line.split("->")
        rhs = rhs.replace('"', "").split()

        if lhs in rules:
            rules[lhs].append(rhs)
        else:
            rules[lhs] = [rhs]

# Level out probabilities
with open(output_file, "w+") as f:
    for lhs, all_rhs in rules.items():
        prob = 1 / len(all_rhs)
        for rhs in all_rhs:
            f.write(f"{prob}\t{lhs}\t{' '.join(rhs)}\n")
