import csv
from pathlib import Path

train_dir = Path('/Users/admin/BodyM/train')
subject_to_photo_map_path = train_dir / 'subject_to_photo_map.csv'
hwg_metadata_path = train_dir / 'hwg_metadata.csv'
combined_data_path = train_dir / 'Combined_Data.csv'

header = ['photo_id', 'height_cm', 'weight_kg', 'BMI']

headline_written = False

with open(subject_to_photo_map_path, mode='r', newline='') as infile_1:
    reader_1 = csv.reader(infile_1)
    subject_to_photo_map = {rows[0]: rows[1] for rows in reader_1 if len(rows) >= 2}
    with open(hwg_metadata_path, mode='r', newline='') as infile_2:
        reader_2 = csv.reader(infile_2)
        hwg_metadata = {rows[0]: rows[2:] for rows in reader_2 if len(rows) >= 3}

        with open(combined_data_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            for subject_id, photo_name in subject_to_photo_map.items():
                if not headline_written:
                    writer.writerow(header)
                    headline_written = True
                if subject_id in hwg_metadata:
                    data = [photo_name] + hwg_metadata.get(subject_id, [])
                    writer.writerow(data)




