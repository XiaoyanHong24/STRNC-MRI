import nbformat
 
# Load the notebook
# notebook_file = "Matching Networks for One Shot Learning.ipynb" #这里放你要转换的ipynb地址
notebook_file = "Model-Agnostic Meta-Learning .ipynb" #这里放你要转换的ipynb地址
with open(notebook_file, "r", encoding="utf-8") as notebook:
    notebook_content = nbformat.read(notebook, as_version=4)
 
# Extract and print Python code
for cell in notebook_content['cells']:
    if cell.cell_type == 'code':
        code = cell['source']
        print(code)