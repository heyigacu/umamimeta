import pandas as pd
import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

parent_dir = os.path.dirname(os.path.abspath(__file__))
forcefield = 'ff14SB'

aa_map = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
    'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

def convert_to_three_letter(sequence):
    return ' '.join([aa_map[aa] for aa in sequence])


def tleap_peptide_model(tleap_in_path, sequence, pdb_path):
    with open(tleap_in_path, 'w') as f:
        f.write(
            f"source leaprc.protein.{forcefield}\n"
            f"mol = sequence {{ {sequence} }}\n"
            f"savepdb mol {pdb_path}\n"
            "quit\n"
        )
    os.system(f'tleap -f {tleap_in_path}')

def rdkit_peptide_model(L_seq, pdb_path):
    L_mol = Chem.rdmolfiles.MolFromFASTA(L_seq, flavor=0)
    L_mol = Chem.AddHs(L_mol)
    AllChem.EmbedMolecule(L_mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(L_mol)
    Chem.MolToPDBFile(L_mol, pdb_path)


def PeptideConstructor_peptide_model(sequence, pdb_path):
    os.system(f"PCcli -s {sequence} -o {pdb_path}")


def pepseq2pdb(input_path=parent_dir+'/umm2024.csv', tleap_in_dir=parent_dir+'/peptide/tleap_in', pdb_dir=parent_dir+'/peptide/pdb', pdbh_dir=parent_dir+'/peptide/pdbh',):
    df = pd.read_csv(input_path, sep='\t', header=0)
    for index, row in df.iterrows():
        if len(row['Sequence']) < 5: 
            print(row['Sequence'])
            sequence = convert_to_three_letter(row['Sequence']) 
            print(sequence)
            tleap_in_path = tleap_in_dir+f"/{row['Sequence']}.in"
            pdb_path = pdb_dir+f"/{row['Sequence']}.pdb"
            pdbh_path = pdbh_dir+f"/{row['Sequence']}.pdb"
            # tleap_peptide_model(tleap_in_path, sequence, pdb_path)
            # rdkit_peptide_model(row['Sequence'], pdb_path)
            # PeptideConstructor_peptide_model(row['Sequence'], pdb_path)
            os.system(f"reduce {pdb_path} > {pdbh_path}")


mgltools_install_dir = '/home/hy/Softwares/MGLTools/'
prepare_ligand4_path = os.path.join(mgltools_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
prepare_receptor4_path = os.path.join(mgltools_install_dir, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py')
pythonsh_path = os.path.join(mgltools_install_dir, 'bin/pythonsh')
reduce_path = "/home/hy/Softwares/amber24/bin/reduce"

def generate_pdbqt(input_path=parent_dir+'/umm2024.csv', column_name='Sequence', pdb_dir=parent_dir+'/peptide/pdb', pdbh_dir=parent_dir+'/peptide/pdbh', pdbqt_save_dir=parent_dir+'/peptide/pdbqt', is_ligand=True):
    ls = []
    df = pd.read_csv(input_path, sep='\t', header=0)
    for index, row in df.iterrows():
        if is_ligand and len(row[column_name]) < 5: 
            continue
        pdbqt_name = row[column_name]
        pdb_path = os.path.join(pdb_dir, pdbqt_name+'.pdb')
        pdbh_path = os.path.join(pdbh_dir, pdbqt_name+'.pdb')
        os.system(f"{reduce_path} {pdb_path} > {pdbh_path}")
        pdbqt_path = os.path.join(pdbqt_save_dir, pdbqt_name+'.pdbqt')
        ls.append(pdbqt_path)
        if is_ligand:
            command = f'cd {pdb_dir}; {pythonsh_path} {prepare_ligand4_path} -l {pdbh_path} -o {pdbqt_path}'
        else:
            command = f'cd {pdb_dir}; {pythonsh_path} {prepare_receptor4_path} -r {pdbh_path} -o {pdbqt_path}'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pdbqt_name}: {e}")

# pepseq2pdb()

# generate_pdbqt(input_path=parent_dir+'/ummpep2024.csv', column_name='Sequence', pdb_dir=parent_dir+'/peptide/pdb', pdbh_dir=parent_dir+'/peptide/pdbh', pdbqt_save_dir=parent_dir+'/peptide/pdbqt', is_ligand=True)
generate_pdbqt(input_path=parent_dir+'/ummrec_all.csv', column_name='receptor', pdb_dir=parent_dir+'/receptor/pdb', pdbh_dir=parent_dir+'/receptor/pdbh', pdbqt_save_dir=parent_dir+'/receptor/pdbqt', is_ligand=False)