import errno
import os
import shutil
import sys
import traceback
import zipfile

common_rvc_subfolder = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(os.path.abspath(__file__))),"../common/"))
if common_rvc_subfolder not in sys.path:
    sys.path.insert(0, common_rvc_subfolder)
from rvc_download_helper import download_file_with_resume

# Converts a string to bytes (for writing the string into a file). Provided for
# compatibility with Python 2 and 3.
def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')


# Outputs the given text and lets the user input a response (submitted by
# pressing the return key). Provided for compatibility with Python 2 and 3.
def GetUserInput(text):
    if sys.version_info[0] == 2:
        return raw_input(text)
    else:
        return input(text)


# Creates the given directory (hierarchy), which may already exist. Provided for
# compatibility with Python 2 and 3.
def MakeDirsExistOk(directory_path):
    try:
        os.makedirs(directory_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# Deletes all files and folders within the given folder.
def DeleteFolderContents(folder_path):
  for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    try:
      if os.path.isfile(file_path):
        os.unlink(file_path)
      else:  #if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Exception in DeleteFolderContents():')
      print(e)
      print('Stack trace:')
      print(traceback.format_exc())


# Creates the given directory, respectively deletes all content of the directory
# in case it already exists.
def MakeCleanDirectory(folder_path):
    if os.path.isdir(folder_path):
        DeleteFolderContents(folder_path)
    else:
        MakeDirsExistOk(folder_path)


# Downloads the file with given GDrive ID to the given destination path.
# https://stackoverflow.com/a/39225039
def DownloadFileFromGDrive(gdrive_id, dest_file_path):
    import requests

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(dest_file_path, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : gdrive_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : gdrive_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, dest_file_path)    
    pass


# Downloads the given URL to a file in the given directory. Returns the
# path to the downloaded file.
# In part adapted from: https://stackoverflow.com/questions/22676
def DownloadFile(url, dest_dir_path):
    file_name = url.split('/')[-1]
    dest_file_path = os.path.join(dest_dir_path, file_name)
    download_file_with_resume(url, dest_file_path, try_resume_bytes=0)
    return dest_file_path

# Unzips the given zip file into the given directory.
def UnzipFile(file_path, unzip_dir_path, overwrite=True):
    zip_ref = zipfile.ZipFile(open(file_path, 'rb'))

    if not overwrite:
        for f in zip_ref.namelist():
            if not os.path.isfile(os.path.join(unzip_dir_path, f)):
                zip_ref.extract(f, path=unzip_dir_path)
            else:
                print('Not overwriting {}'.format(f))
    else:
        zip_ref.extractall(unzip_dir_path)
    zip_ref.close()


# Creates a zip file with the contents of the given directory.
# The archive_base_path must not include the extension .zip. The full, final
# path of the archive is returned by the function.
def ZipDirectory(archive_base_path, root_dir_path):
  #  return shutil.make_archive(archive_base_path, 'zip', root_dir_path) # THIS WILL ALWAYS HAVE ./ FOLDER INCLUDED
	with zipfile.ZipFile(archive_base_path+'.zip', "w", compression=zipfile.ZIP_DEFLATED) as zf:
		  base_path = os.path.normpath(root_dir_path)
		  for dirpath, dirnames, filenames in os.walk(root_dir_path):
		      for name in sorted(dirnames):
		          path = os.path.normpath(os.path.join(dirpath, name))
		          zf.write(path, os.path.relpath(path, base_path))
		      for name in filenames:
		          path = os.path.normpath(os.path.join(dirpath, name))
		          if os.path.isfile(path):
		              zf.write(path, os.path.relpath(path, base_path))

		  return archive_base_path+'.zip'


# Downloads a zip file and directly unzips it.
def DownloadAndUnzipFile(url, archive_dir_path, unzip_dir_path, overwrite=True):
    archive_path = DownloadFile(url, archive_dir_path)
    UnzipFile(archive_path, unzip_dir_path, overwrite=overwrite)
