#!/usr/bin/env python3

import rosbag
import csv
import os
import sys

def bag_to_csv(bag_path, output_dir):
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages():
            csv_file = os.path.join(output_dir, f"{topic.replace('/', '_')[1:]}.csv")
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([t.to_sec(), msg])

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: bag_to_csv.py <input_bag> <output_dir>")
#         sys.exit(1)
#     bag_to_csv(sys.argv[1], sys.argv[2])
