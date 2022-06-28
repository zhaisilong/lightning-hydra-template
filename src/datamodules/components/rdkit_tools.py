from rdkit import Chem

def pep2smi(pep: str):
    return Chem.MolToSmiles(Chem.MolFromFASTA(pep))

def smi2pep(smi: str) -> str:
    """不能用

    Notes:
        - 默认正交

    Args:
        smi: SMILEs

    Returns:
        str: 氨基酸序列
    """
    return Chem.MolToFASTA(Chem.MolFromSmiles(smi))