# ![image.png](attachment:image.png)

# +
import os
import glob
import subprocess

# CSV 파일 경로 설정
csv_directory = '/home/work/DLmath/seungeun/tab/suite/cc18_num/2'

# 실행할 스크립트 경로 설정
fashionmnist_script = 'fashionmnist_classification_s.py'
diabetes_tab_script = 'diabetes_tab_s.py'
diabetes_inv_script = 'diabetes_inv_s.py'
final_script = 'FiFinal_csv_ModelFigSave_mean_mmd_elbo_diabetes_interpretation_val_3.py'

# CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

# 현재 폴더에 FashionMNIST 스크립트 실행하여 weight 저장
print("Running FashionMNIST classification...")
subprocess.run(['python', fashionmnist_script], check=True)

csv_files = csv_files[::-1][4:]

for csv_file in csv_files:
    # CSV 파일 이름에서 폴더 이름 생성
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_folder = os.path.join(csv_directory, csv_name)

    # 해당 CSV 이름에 맞는 폴더 생성
    os.makedirs(csv_name, exist_ok=True)

    # Step 1: diabetes_tab.py 실행
    print(f"Running diabetes_tab_s.py for {csv_file}...")
    subprocess.run(
        ['python', diabetes_tab_script, '--csv', csv_file, '--csv_name', csv_name],
        check=True
    )

    # Step 2: diabetes_inv.py 실행
    print(f"Running diabetes_inv_s.py for {csv_file}...")
    subprocess.run(
        ['python', diabetes_inv_script, '--csv', csv_file, '--csv_name', csv_name],
        check=True
    )

    # Step 3: FiFinal_csv_ModelFigSave_mean_mmd_elbo_diabetes_interpretation_val_3.py 200번 실행
    for index in range(1, 11):
        print(f"Running final script ({index}/10) for {csv_file}...")
        subprocess.run(
            ['python', final_script, '--csv', csv_file, '--csv_name', csv_name, '--index', str(index)],
            check=True
        )

print("All processes completed!")
# +
# import numpy as np

# # Example arrays
# arr = np.random.rand(148, 1,28,28, 5)  # Shape: (148, 19, 5)
# index_array = np.random.randint(0, 5, 148)  # Shape: (148,)

# # Use advanced indexing to select features,  ㅝㅡㅏ, 
# result = arr[np.arange(148), :,:,:, index_array]

# print(result.shape)  # Output: (148, 19, 1)

