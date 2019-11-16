import csv
import os
from datetime import datetime
import sys


class DataFile:
    import csv
    import sys

    def __init__(self, path):
        self.file_path = path
        self.data = None
        self.init_data()

    def set_file_path(self, path):
        self.file_path = path

    def init_data(self):
        print("Init with : ", self.file_path)
        file = open(self.file_path, 'r')
        self.data = csv.DictReader(file)

    def get_highest_list_year(self):
        print("Looking for newest list year...")
        best = 1
        for row in self.data:
            date = int(row["ListYear"])
            if date > best:
                best = date
        return best


dir = os.getcwd() + "\\" + "data.csv"
dt = DataFile(path=dir)
print(dt.get_highest_list_year())
