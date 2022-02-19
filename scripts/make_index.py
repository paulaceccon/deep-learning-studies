from pathlib import Path
import nbformat as nbf
import os

def to_title(string: str) -> str:
    return string.replace("_", " ").title()


def get_notebooks(root_dir: str) -> list[Path]:
    root_dir = Path(root_dir)
    return [f for f in root_dir.glob('**/*.ipynb') if f.is_file() and ".ipynb_checkpoints" not in str(f.parent)]


def create_notebook(notebooks: list[Path]):
    project = os.getcwd()
    badge = f"<a href=\"https://colab.research.google.com/github/paulaceccon/{project}/blob/main/index.ipynb\" target=\"_parent\" style=\"float: left;\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
    title_cell = nbf.v4.new_markdown_cell("### Notebooks")
    badge_cell = nbf.v4.new_markdown_cell(badge)

    for notebook in notebooks:
        levels = str(notebook).split("/")[1:-1]
        items = []
        i = 0
        for i, level in enumerate(levels):
            name = to_title(level)
            pad = "\t"*i
            items.append(f"{pad} - {name}")

        if i > 0:
            i += 1
        pad = "\t"*i

        name = to_title(notebook.stem)
        items.append(f"{pad}- [{name}]({notebook})")
        items = "\n".join(items)
        items_cell = nbf.v4.new_markdown_cell(items)

    nb = nbf.v4.new_notebook()
    nb['cells'] = [title_cell,
                   badge_cell,
                   items_cell]
    nbf.write(nb, 'index.ipynb')


if __name__ == "__main__":
    notebooks = get_notebooks('./notebooks')
    create_notebook(notebooks)

