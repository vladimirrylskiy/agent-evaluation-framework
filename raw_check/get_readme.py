from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='mcemri/MAD', filename='README.md', repo_type='dataset')
shutil.copy(p, 'raw_check/MAD_README.md')
print('README saved to raw_check/MAD_README.md')
