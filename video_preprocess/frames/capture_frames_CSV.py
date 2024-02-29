import subprocess
from tqdm import tqdm

with open('Datasets/CSV/test_split.txt', 'r') as f:
    data_list = [line.strip().split() for line in f.readlines()]

with open('Datasets/CSV/train_split.txt', 'r') as f:
    data_list += [line.strip().split() for line in f.readlines()]

for name, no in tqdm(data_list):
    output_dir = f'/mnt/data240/CSV/videos_compressed/{no}/{name}'
    input_dir = f'/mnt/data240/CSV/videos/{no}/{name}'
    cmd = f"/usr/bin/ffmpeg -i {input_dir} -s 320x180 -pix_fmt yuv420p -crf 18 -tune fastdecode -loglevel error {output_dir} -y"
    subprocess.run(cmd.split())
