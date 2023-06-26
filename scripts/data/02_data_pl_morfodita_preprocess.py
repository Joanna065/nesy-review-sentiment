from pathlib import Path

from src.settings import DATA_DIR

HEADER_LINES = [
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    '<!DOCTYPE chunkList SYSTEM "ccl.dtd">\n',
]


def get_proper_ccl(path: Path, split: str) -> list[str]:
    with path.open(mode='r') as f:
        lines = f.readlines()

    doc_id = int(str(path.stem).split('.')[0])

    new_lines = []
    new_lines.extend(HEADER_LINES)

    lines = lines[3:]

    sent_count = 0
    token_count = 0
    for line in lines:
        if line == '<chunk id="1" type="p">\n':
            continue
        if 'cesAna' in line:
            continue
        if 'lex' in line and 'disamb="1"' not in line:
            continue

        if 'type="s"' in line:
            sent_count += 1
            token_count = 0
            line = f'<chunk id="{sent_count}">\n'

        if '<tok>' in line:
            token_count += 1
            line = f'<tok instance="{split}.d{str(doc_id).zfill(3)}.s{str(sent_count).zfill(3)}.t{str(token_count).zfill(4)}">\n'

        new_lines.append(line)

    return new_lines


if __name__ == '__main__':
    splits = ['train', 'dev', 'test']
    dataset = 'polemo2'

    for split in splits:
        ccl_dir = DATA_DIR.joinpath(dataset, 'ccl', f'all_text_{split}')

        for path in ccl_dir.iterdir():
            new_ccl_lines = get_proper_ccl(path, split=split)
            with ccl_dir.joinpath(path.name).open(mode='w') as f:
                f.writelines(new_ccl_lines)

    dataset = 'klej_ar'
    for split in splits:
        ccl_dir = DATA_DIR.joinpath(dataset, 'ccl', f'{split}')

        for path in ccl_dir.iterdir():
            new_ccl_lines = get_proper_ccl(path, split=split)
            with ccl_dir.joinpath(path.name).open(mode='w') as f:
                f.writelines(new_ccl_lines)
