from typing import Optional


def seq_padding(seq: str,
                length: int,
                direction: str = 'right',
                filling='[PAD]',
                truncate_length: Optional[int] = None,
                split: bool = True):
    tmp_filling = '神'

    if direction == 'right':
        seq_l = list(seq.ljust(length, tmp_filling))
    else:
        seq_l = list(seq.rjust(length, tmp_filling))

    rt = [filling if w == tmp_filling else w for w in seq_l]

    if truncate_length:
        rt = rt[:truncate_length]

    if split:
        return rt
    else:
        return ' '.join(rt)


if __name__ == '__main__':
    print(seq_padding('456', 10, truncate_length=9))
