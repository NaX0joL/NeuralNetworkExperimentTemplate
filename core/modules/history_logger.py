# not tested!!!



class HistoryLogger():
    def __init__(self) -> None:
        self.history_data_log: dict[list] = {}
        return
    
    def insert_data(self, data_name:str, data_value) -> None:
        if data_name in self.history_data_log.keys():
            self.history_data_log[data_name].append(data_value)
        else:
            self.history_data_log[data_name] = [data_value]
        return
        
    def get_data(self, data_name:str) -> list:
        data = self.history_data_log[data_name]
        return data
    
    def retrieve_all_logs(self) -> dict:
        all_logs = self.history_data_log
        return all_logs
    
    def empty_log(self) -> None:
        self.history_data_log: dict[list] = {}
        return



if __name__ == "__name__":
    pass