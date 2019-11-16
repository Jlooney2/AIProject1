import pandas

class Preprocessor:
    
    def __init__(self):
        self.data_columns = ['SerialNumber','ListYear','Town','Address','AssessedValue','SaleAmount','PropertyType']

    def process_file(self, filepath):
        file = pandas.read_csv(filepath)
        processed_file = file[self.data_columns]
        processed_file.to_csv(f'processed.csv', index=False)
  