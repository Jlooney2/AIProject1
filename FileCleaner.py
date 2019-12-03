import fileinput


def clean(filename):
    seen = set()  # set for fast O(1) amortized lookup

    duplicates = 0
    zeros = 0
    for line in fileinput.FileInput(filename, inplace=1):
        if line in seen:
            duplicates += 1
            continue  # skip duplicate
        split_line = line.split(',')
        skip = False
        for item in split_line:
            if item == '0' or item == '0.0':
                zeros += 1
                skip = True
                continue
        if skip:
            continue
        seen.add(line)
        print(line, end='')
    print("File Cleaner removed " + str(duplicates) + " duplicate rows and " + str(zeros) + " zero value rows")
