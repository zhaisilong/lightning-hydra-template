from rdkit import Chem

def pep2smi(pep: str):
    return Chem.MolToSmiles(Chem.MolFromFASTA(pep))