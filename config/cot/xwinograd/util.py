import argparse
import yaml

def _doc_to_text(doc, cot=False):
    prepend = "Fill in the blank: "
    text = prepend + f"{doc['sentence']}\n(A) {doc['option1']}\n(B) {doc['option2']}\n"
    if cot:
        text += "Let's think step by step."
    return text

def doc_to_text(doc):
    return _doc_to_text(doc, cot=False)

def doc_to_text_cot(doc):
    return _doc_to_text(doc, cot=True)

def doc_to_target(doc):
    choices = ['(A)', '(B)']
    return choices[int(doc['answer']) - 1]


LANGUAGES = ["en", "fr", "jp", "pt", "ru", "zh"]


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for cot in (False, True):
        for lang in LANGUAGES:
            cot_string = 'cot' if cot else 'generate'
            task_name = f"xwinograd_{cot_string}_{lang}"
            try:
                with open(
                    f"{output_dir}/{task_name}.yaml", "w" if overwrite else "x", encoding="utf-8"
                ) as f:
                    f.write("# Generated by util.py\n")
                    yaml.dump(
                        {
                            "include": f"xwinograd_{cot_string}_common_yaml",
                            "dataset_name": lang,
                            "task": task_name,
                        },
                        f,
                    )
            except FileExistsError:
                err.append(f'{task_name}.yaml')

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()