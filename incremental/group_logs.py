from tbparse import SummaryReader
import os
import shutil

first_log_dir = "initial_pretrain_jz/run1"
second_log_dir = "initial_pretrain_jz/run2"
merged_log_dir = "initial_pretrain_jz/merged_logs"

reader1 = SummaryReader(first_log_dir)
reader2 = SummaryReader(second_log_dir)

max_step = reader1.scalars["step"].max()
print(f"Shifting second run by {max_step +1} steps")

if os.path.exists(merged_log_dir):
    shutil.rmtree(merged_log_dir)
os.makedirs(merged_log_dir, exist_ok=True)

# Copy first log as-is
for file in os.listdir(first_log_dir):
    shutil.copy(os.path.join(first_log_dir, file), merged_log_dir)

# Re-log second run with shifted steps
import tensorflow as tf
import pandas as pd

shifted = reader2.scalars.copy()
shifted["step"] += max_step + 1

# Re-log with TensorFlow summary writer
writer = tf.summary.create_file_writer(merged_log_dir)

with writer.as_default():
    for row in shifted.itertuples():
        tf.summary.scalar(row.tag, row.value, step=row.step)
    writer.flush()

print("Merged log created at:", merged_log_dir)