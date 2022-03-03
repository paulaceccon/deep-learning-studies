import nbformat as nbf
import os


def to_title(string: str) -> str:
    return string.replace("_", " ").title()


def create_notebook(root_dir: str):
    project = os.getcwd()
    badge = f"<a href=\"https://colab.research.google.com/github/paulaceccon/{project}/blob/main/index.ipynb\" target=\"_parent\" style=\"float: left;\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
    title_cell = nbf.v4.new_markdown_cell("### Notebooks")
    badge_cell = nbf.v4.new_markdown_cell(badge)

    items = []
    indent = "\t"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(exclude in dirpath for exclude in ["logs", ".ipynb_checkpoints"]):
            continue
        directory_level = dirpath.replace(root_dir, "")
        directory_level = directory_level.count(os.sep)

        pad = indent * directory_level
        title = to_title(os.path.basename(dirpath))
        items.append(f"{pad}- {title}")
        print(f"{pad}- {title}")

        for f in filenames:
            if f.endswith(".ipynb"):
                pad = indent * (directory_level + 1)
                title = to_title(f.split(".")[0])
                link = os.path.join(dirpath, f)
                items.append(f"{pad}- [{title}]({link})")
                print(f"{pad}- {title}")

    items = "\n".join(items)
    print(items)
    items_cell = nbf.v4.new_markdown_cell(items)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [title_cell,
                   badge_cell,
                   items_cell]
    nbf.write(nb, "index.ipynb")


if __name__ == "__main__":
    create_notebook("./notebooks")
