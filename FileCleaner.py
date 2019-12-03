import fileinput


def clean(filename):
    seen = set()  # set for fast O(1) amortized lookup

    for line in fileinput.FileInput(filename, inplace=1):
        if line in seen: continue  # skip duplicate
        split_line = line.split(',')
        for item in split_line:
            if item == '0' or item == '0.0': continue
        seen.add(line)
        print(line, end='')
