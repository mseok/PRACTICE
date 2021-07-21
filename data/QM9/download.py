import os
import subprocess

PATH = "xyz_files"
if not os.path.exists(PATH):
    os.mkdir(PATH)
url = "https://ndownloader.figshare.com/files/3195389"
fn = "dsgdb9nsd.xyz.tar.bz2"
command = f"wget {url} -O {fn}"
subprocess.run(command.split())
command = f"tar -xvf {fn} -C {PATH}"
subprocess.run(command.split())
