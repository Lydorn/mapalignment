import tensorflow as tf


class TFRecordShardWriter:
    def __init__(self, filepath_format, max_records_per_shard):
        self.filepath_format = filepath_format
        self.max_records_per_shard = max_records_per_shard
        self.current_shard_record_count = 0  # To know when to switch to a new file
        self.current_shard_count = 0  # To know how to name the record file
        self.writer = None
        self.create_new_shard_writer()

    def create_new_shard_writer(self):
        filename = self.filepath_format.format(self.current_shard_count)
        self.writer = tf.python_io.TFRecordWriter(filename)
        self.current_shard_count += 1

    def write(self, serialized_example):
        self.current_shard_record_count += 1
        if self.max_records_per_shard < self.current_shard_record_count:
            self.create_new_shard_writer()
            self.current_shard_record_count = 1
        self.writer.write(serialized_example)

    def close(self):
        self.writer.close()
