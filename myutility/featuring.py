import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 분자 전체 디스크립터
        features = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumRings': Descriptors.RingCount(mol),
            'TPSA': Descriptors.TPSA(mol),
        }

        # 원자 기반 집계 피처
        num_atoms = mol.GetNumAtoms()
        # atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        # symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        nums_h_atom = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
        radicals = [atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()]
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        # orbitals = [atom.GetHybridization() for atom in mol.GetAtoms()]

        # 결합 기반 집계 피처
        # [for bond in mol.GetBonds()]
        # bond.GetBeginAtomIdx() 와 bond.GetEndAtomIdx() 를 이용해서 연결 표현
        # bond.GetBondType()
        # bond.GetIsConjugated()
        # bond.IsInRing()

        features['NumAtoms'] = num_atoms
        features['MeanDegree'] = np.mean(atom_degrees) if atom_degrees else 0
        features['MeanCharge'] = np.mean(charges) if charges else 0
        features['AromaticProportion'] = num_aromatic_atoms / num_atoms if num_atoms > 0 else 0
        features['FreeElectrons'] = np.sum(radicals)
        features['HydrogenNumber'] = np.sum(nums_h_atom)

        return features

    except:
        return None