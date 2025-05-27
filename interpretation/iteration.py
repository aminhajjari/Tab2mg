import os
import glob
import subprocess

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--csv_dir', type=str, required=True, help='Path to the csv directory')
parser.add_argument('--num_classes', type=int, required=True, help='# of classes')
args = parser.parse_args()

csv_directory = args.csv_dir
num_classes = args.num_classes


image_script = 'default_image_classification.py'
tab_script = 'tab_emb.py'
inv_script = 'inverse_emb.py'
final_script = 'dualshap.py'

csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

print("Running Image classification script...")
subprocess.run(['python', fashionmnist_script, '--num_classes', num_classes], check=True)

for csv_file in csv_files:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_folder = os.path.join(csv_directory, csv_name)

    os.makedirs(csv_name, exist_ok=True)

    # Step 1.
    print(f"Running diabetes_tab_s.py for {csv_file}...")
    subprocess.run(
        ['python', tab_script, '--csv', csv_file, '--csv_name', csv_name],
        check=True
    )

    # Step 2.
    print(f"Running diabetes_inv_s.py for {csv_file}...")
    subprocess.run(
        ['python', inv_script, '--csv', csv_file, '--csv_name', csv_name],
        check=True
    )

    # Step 3.
    for index in range(1, 11):
        print(f"Running final script ({index}/10) for {csv_file}...")
        subprocess.run(
            ['python', final_script, '--csv', csv_file, '--csv_name', csv_name, '--index', str(index)],
            check=True
        )

print("All processes completed!")

