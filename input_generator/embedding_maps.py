from typing import Dict

embedding_map_fivebead = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}

dict_HP = {
    'ALA': 'H',  # A
    'CYS': 'H',  # C
    'ASP': 'P',  # D
    'GLU': 'P',  # E
    'PHE': 'H',  # F
    'GLY': 'H',  # G
    'HIS': 'P',  # H
    'ILE': 'H',  # I
    'LYS': 'P',  # K
    'LEU': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'P',  # N
    'PRO': 'H',  # P
    'GLN': 'P',  # Q
    'ARG': 'P',  # R
    'SER': 'P',  # S
    'THR': 'P',  # T
    'VAL': 'H',  # V
    'TRP': 'H',  # W
    'TYR': 'P',  # Y
}


dict_HPNX = {
    'ALA': 'H',  # A
    'CYS': 'H',  # C
    'ASP': 'N',  # D
    'GLU': 'N',  # E
    'PHE': 'H',  # F
    'GLY': 'X',  # G
    'HIS': 'P',  # H
    'ILE': 'H',  # I
    'LYS': 'P',  # K
    'LEU': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'X',  # N
    'PRO': 'X',  # P
    'GLN': 'X',  # Q
    'ARG': 'Y',  # R
    'SER': 'X',  # S
    'THR': 'X',  # T
    'VAL': 'H',  # V
    'TRP': 'H',  # W
    'TYR': 'Y'   # Y
}


dict_hHPNX = {
    'ALA': 'h',  # A
    'CYS': 'H',  # C
    'ASP': 'N',  # D
    'GLU': 'N',  # E
    'PHE': 'H',  # F
    'GLY': 'X',  # G
    'HIS': 'P',  # H
    'ILE': 'H',  # I
    'LYS': 'P',  # K
    'LEU': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'X',  # N
    'PRO': 'X',  # P
    'GLN': 'X',  # Q
    'ARG': 'Y',  # R
    'SER': 'X',  # S
    'THR': 'X',  # T
    'VAL': 'h',  # V
    'TRP': 'H',  # W
    'TYR': 'Y'   # Y
}

dict_YhHX = {
    'ALA': 'H',  # A
    'CYS': 'H',  # C
    'ASP': 'Y',  # D
    'GLU': 'Y',  # E
    'PHE': 'H',  # F
    'GLY': 'X',  # G
    'HIS': 'Y',  # H
    'ILE': 'H',  # I
    'LYS': 'Y',  # K
    'LEU': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'Y',  # N
    'PRO': 'X',  # P
    'GLN': 'X',  # Q
    'ARG': 'Y',  # R
    'SER': 'X',  # S
    'THR': 'X',  # T
    'VAL': 'H',  # V
    'TRP': 'H',  # W
    'TYR': 'Y'   # Y
}

final_letter_map = {
    'H' : 1,
    'P' : 2,
    'N' : 3,
    'X' : 4,
    'Y' : 5,
    'h' : 6,
}

extra_5bead = {
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}

embedding_HP = { key : final_letter_map[val] for key, val in dict_HP.items() }
for key, val in extra_5bead.items():
    embedding_HP[key] = val

embedding_YhHX = { key : final_letter_map[val] for key, val in dict_YhHX.items() }
for key, val in extra_5bead.items():
    embedding_YhHX[key] = val

embedding_HPNX = { key : final_letter_map[val] for key, val in dict_HPNX.items() }
for key, val in extra_5bead.items():
    embedding_HPNX[key] = val

embedding_hHPNX = { key : final_letter_map[val] for key, val in dict_hHPNX.items() }
for key, val in extra_5bead.items():
    embedding_hHPNX[key] = val


class CGEmbeddingMap(dict):
    """
    General class for defining embedding maps as Dict
    """

    def __init__(self, embedding_map_dict: Dict[str, int]):
        for k, v in embedding_map_dict.items():
            self[k] = v


class CGEmbeddingMapFiveBead(CGEmbeddingMap):
    """
    Five-bead embedding map defined by:
        - N : backbone nitrogen
        - CA : backbone alpha carbon (specialized for glycing)
        - C : backbone carbonyl carbon
        - O : backbone carbonyl oxygen
        - CB : residue-specific beta carbon
    """

    def __init__(self):
        super().__init__(embedding_map_fivebead)


class CGEmbeddingMapCA(CGEmbeddingMap):
    """
    One-bead embedding map defined by:
        - CA : backbone alpha carbon, carrying aminoacid identity
    """

    def __init__(self):
        ca_dict = {key: emb for key, emb in embedding_map_fivebead.items() if emb <= 20}
        super().__init__(ca_dict)

class CGEmbeddingMapHP(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_HP)

class CGEmbeddingMapYhHX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_YhHX)

class CGEmbeddingMapHPNX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_HPNX)

class CGEmbeddingMaphHPNX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_hHPNX)

all_residues = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def embedding_fivebead(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_fivebead[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_fivebead["GLY"]
        else:
            atom_type = embedding_map_fivebead[name]
    elif name == "CB":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_ca(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_HP(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_HP[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_HPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_HPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_YhHX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_YhHX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_hHPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_hHPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type
