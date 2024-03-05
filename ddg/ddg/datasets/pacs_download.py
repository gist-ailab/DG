import os
import gdown
from zipfile import ZipFile
def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    # if dst.endswith(".tar.gz"):
    #     tar = tarfile.open(dst, "r:gz")
    #     tar.extractall(os.path.dirname(dst))
    #     tar.close()

    # if dst.endswith(".tar"):
    #     tar = tarfile.open(dst, "r:")
    #     tar.extractall(os.path.dirname(dst))
    #     tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

def download_pacs():
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = "/ailab_mat/dataset/DDG/PACS/PACS"

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

down