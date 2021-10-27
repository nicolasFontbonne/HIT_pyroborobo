import os
from datetime import datetime


now = datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M-%S-%f")

folder = f'logs/{date}/'
os.mkdir(folder)

repet = 32

for i in range(repet):
    os.system(f"python hit.py {folder}")