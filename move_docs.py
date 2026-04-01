import os
import shutil
import glob
import subprocess

base_dir = "INTERNS_ASSESSMENT/Personalized-Speech-Learning-using-TTS"
docs_dir = os.path.join(base_dir, "docs")

for f in glob.glob(os.path.join(base_dir, "*.md")):
    basename = os.path.basename(f)
    if basename == "dfd.md":
        dest_name = "diagram.md"
    else:
        dest_name = basename
    
    shutil.move(f, os.path.join(docs_dir, dest_name))

subprocess.run(["git", "add", "-A"])
subprocess.run(["git", "commit", "-m", "Moved all root markdown files into docs folder and properly renamed dfd to diagram"])

os.remove(__file__)
