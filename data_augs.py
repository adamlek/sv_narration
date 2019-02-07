def extract_lines(line_file):
    with open(line_file) as lf:
        return [x for x in lf.read().split('\n') if x]


def format_lines_forstagger(lines):
    all_lines = []
    for line in lines:
        all_lines.append([])
        line = line.split()
        for word in line:
            if word == '…':
                all_lines[-1].append('...')
            elif word.endswith('…'):
                all_lines[-1].append(word[:-1])
                all_lines[-1].append('...')
            # elif word == '<NC>':
            #     continue
            else:
                all_lines[-1].append(word)
    
    return [' '.join(x) for x in all_lines]


if __name__ == '__main__':
    line_file = '/home/adam/git/dialogue_structure_v2/all_data/dialogue_lines.txt'
    lines = extract_lines(line_file)

    # create data for stagger
    lines = format_lines_forstagger(lines)
    for line in lines:
        print(line)
        print()
