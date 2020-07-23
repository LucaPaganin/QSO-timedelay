from pathlib import Path
import shutil
import os

srcdir = Path(os.getenv("HOME")) / "OnlyLocalFiles/Mendeley_bibfiles"
srcfile = srcdir / "QSO time delay.bib"

dstfile = Path('.') / "QSO_timedelay.bib"

if dstfile.exists():
    dstfile.unlink()
shutil.copy(srcfile, dstfile)
