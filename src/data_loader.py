def read_and_split_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    paragraphs = text.split("\n")
    # Filter out any empty paragraphs or undesired entries
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 0]
    return paragraphs
