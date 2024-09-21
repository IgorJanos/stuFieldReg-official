from project.base.logging import Logger
from pathlib import Path

def default_value_formatter(value):
    if (isinstance(value, int)):
        return f"{value}"
    return "{:.10f}".format(value)




class CSVWriter:
    def __init__(
        self, 
        filename: Path, 
        separator=",",
        formatter=default_value_formatter
    ):
        self.filename = filename
        self.file = None
        self.lines = 0
        self.separator = separator
        self.formatter = formatter

    def open(self):
        self.lines = 0
        self.file = open(self.filename.as_posix(), "w")

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, items):
        if self.file is not None:
            if self.lines == 0:
                # CSV header
                line = self.separator + self.separator.join(list(items.keys()))
                self.file.write(line + "\n")

            values = [ self.formatter(items[k]) for k in items.keys() ]
            line = self.separator.join(values)
            self.file.write(line + "\n")
            self.file.flush()
            self.lines += 1
        


class CSVLogger(Logger):
    def __init__(
        self, 
        filename: Path, 
        separator=",",
        formatter=default_value_formatter
    ):
        self.writer = CSVWriter(filename, separator, formatter)

    def on_training_start(self):
        self.writer.open()

    def on_training_end(self):
        self.writer.close()

    def on_epoch_end(self, epoch, stats):
        
        # Insert also epoch        
        epoch_stats = {
            "epoch": int(epoch)
        }
        epoch_stats.update(stats)
        
        self.writer.write(epoch_stats)

