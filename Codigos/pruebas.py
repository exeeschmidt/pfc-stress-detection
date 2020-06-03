from tqdm import tqdm
from colorama import Fore
import time

bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN)
with tqdm(total=10, unit='frame', desc="Frames", bar_format=bar_format) as barra2:
    for num in range(10):
        barra2.update(1)
        time.sleep(0.5)
