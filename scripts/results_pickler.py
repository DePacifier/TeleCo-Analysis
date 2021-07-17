from pickle import dump, load, HIGHEST_PROTOCOL

class ResultPickler():
    def __init__(self) -> None:
        self.data = {}
    
    def add_data(self,name:str, data) -> None:
        self.data[name] = data
    
    def save_data(self,file_name:str) -> None:
        with open(file_name, 'wb') as handle:
            dump(self.data, handle, protocol=HIGHEST_PROTOCOL)
    
    def load_data(self,file_name:str) -> None:
        with open(file_name, 'rb') as handle:
            self.data = load(handle)

    def get_data(self) -> dict:
        return self.data