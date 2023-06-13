def find_last_idx(string, sub):
    index_of_occurrences = []
    current_index = 0
    while True:
        current_index = string.find(sub, current_index)
        if current_index == -1:
            return index_of_occurrences[-1]
        else:
            index_of_occurrences.append(current_index)
            current_index += len(sub)

def find_words_last_idx(string, subs):
    ret = -1
    for sub in subs:
        idx = find_last_idx(string, sub)
        ret = max(ret, idx)
    return ret
