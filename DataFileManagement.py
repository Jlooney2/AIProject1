import csv


class DataFile:
    import csv

    def __init__(self):
        self.file_path = None
        self.data = dict

    def set_file_path(self, path):
        self.file_path = path

    def read_data(self):
        with open(self.file_path, mode='r') as in_file:
            reader = csv.reader(in_file, delimiter=',')
            line_count = 0
            self.data.fromkeys(reader[0])

            for key in self.data.keys():
                self.data[key] = []

            for row in reader[1:]:
                print(row)

