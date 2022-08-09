import nbformat as nbf
import os


def to_title(string: str) -> str:
    """
    Given a string, format it to title.

    Args:
        string: string to be formatted.
    Returns:
        string as title.
    """
    keep_uppercase = ["cnn", "gan"]
    str_parts = string.split("_")
    for i, part in enumerate(str_parts):
        if any(keep_upper in part for keep_upper in keep_uppercase):
            str_parts[i] = part.upper()
        else:
            str_parts[i] = part.title()
    return " ".join(str_parts)


def create_notebook(root_dir: str) -> None:
    """
    Create a notebook based on the .ipynb files found.

    Args:
        root_dir: where to look for .ipynb files.
    """
    project = os.getcwd()
    badge = f"<a href=\"https://colab.research.google.com/github/paulaceccon/{project}/blob/main/index.ipynb\" target=\"_parent\" style=\"float: left;\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
    title_cell = nbf.v4.new_markdown_cell("### Notebooks")
    badge_cell = nbf.v4.new_markdown_cell(badge)

    items = []
    indent = "\t"
    for dir_path, dir_names, filenames in os.walk(root_dir):
        # Skip unwanted files
        if any(exclude in dir_path for exclude in ["logs", ".ipynb_checkpoints"]):
            continue

        directory_level = dir_path.replace(root_dir, "")
        directory_level = directory_level.count(os.sep)

        # Create category items
        pad = indent * directory_level
        title = to_title(os.path.basename(dir_path))
        items.append(f"{pad}- {title}")
        print(f"{pad}- {title}")

        # Create .ipynb items with link
        for f in filenames:
            if f.endswith(".ipynb"):
                pad = indent * (directory_level + 1)
                title = to_title(f.split(".")[0])
                link = os.path.join(dir_path, f)
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
