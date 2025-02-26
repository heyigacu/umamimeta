import pandas as pd
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))

hippos = "/home/hy/Softwares/miniconda/envs/hippos/bin/hippos"


def extract_best_conformation(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    best_score = float('inf')
    best_model = []
    current_model = []
    recording = False
    for line in lines:
        if line.startswith("MODEL"):
            recording = True
            current_model = []
        elif "ENDMDL" in line:
            recording = False
            score_line = [l for l in current_model if "REMARK VINA RESULT" in l]
            if score_line:
                score = float(score_line[0].split()[3])
                if score < best_score:
                    best_score = score
                    best_model = current_model
        if recording:
            current_model.append(line)
    return best_model, best_score

def get_vina_result(vina_out_path, save_name):
    best_model, best_score = extract_best_conformation(vina_out_path)
    project_dir = os.path.dirname(vina_out_path)
    save_path_name = project_dir+'/'+save_name
    # with open(save_path_name+'.pdbqt', 'w') as f:
    #     f.writelines(best_model)
    # os.system(f'cd {project_dir}; obabel -ipdbqt {save_name}.pdbqt -opdb -O {save_name}.pdb')

    return best_score, save_path_name+'.pdb'

def get_adcp_result(project_dir, save_name):
    import re
    best_pdb = project_dir+f'/{save_name}_ranked_1.pdb'
    log = project_dir+f'/{save_name}.log'
    with open(log, "r") as file:
        content = file.read()
    match = re.search(r'^\s*(\d+)\s+(-?\d+\.\d+)', content, re.MULTILINE)
    if match:
        best_score = match.group(2)
    save_path_name = project_dir+'/'+save_name
    return best_score, best_pdb

import shutil


def batct_plif(rec_input_path=parent_dir+'/ummrec_all.csv', pep_input_path=parent_dir+'/ummpep2024.csv', 
               trg_dir=parent_dir+'/receptor/trg', dock_dir=parent_dir+'/dock', rec_mol2_dir=parent_dir+'/receptor/mol2',
               rec_pdbqt_dir=parent_dir+'/receptor/pdbqt', pep_pdbqt_dir=parent_dir+'/peptide/pdbqt',):
    df_rec = pd.read_csv(rec_input_path, sep='\t', header=0)
    df_pep = pd.read_csv(pep_input_path, sep='\t', header=0)
    df_pep = df_pep[df_pep['lenth'] < 16]
    print(df_pep.shape)
    dic = {}
    for index_rec, row_rec in df_rec.iterrows():
        receptor_pdbqt_path = f"{rec_pdbqt_dir}/{row_rec['receptor']}.pdbqt"
        receptor_mol2_path = f"{rec_mol2_dir}/{row_rec['receptor']}.pdbqt"
        # plif_dir = dock_dir+f"/{row_rec['pocket']}/plif/"
        # if not os.path.exists(plif_dir): os.makedirs(project_dir)
        temp_arr = []
        ligand_paths = []
        erros = []
        for index_pep, row_pep in df_pep.iterrows():
            project_dir = dock_dir+f"/{row_rec['pocket']}/{row_pep['Sequence']}/"
            project_name = f"{row_rec['pocket']}_{row_pep['Sequence']}"
            pep_pdbqt_path = f"{pep_pdbqt_dir}/{row_pep['Sequence']}.pdbqt"
            # ADCP
            if 16 > len(row_pep['Sequence']) >= 5: 
                try:
                    best_score, PDB_path = get_adcp_result(project_dir, project_name)
                except:
                    best_score, PDB_path = None, None
                temp_arr.append(best_score)
            # VINA
            elif len(row_pep['Sequence']) < 5:
                if not os.path.exists(project_dir): os.makedirs(project_dir)
                dock_out_path = f"{project_dir}/{project_name}_out.pdbqt"
                try:
                    best_score, PDB_path = get_vina_result(dock_out_path, project_name)
                except:
                    erros.append(project_name)
                    best_score, PDB_path = None, None
                temp_arr.append(best_score)
            ligand_paths.append(PDB_path)
        print(erros)
        df_pep[row_rec['receptor']] = temp_arr


    df_pep.to_csv('plif.txt', sep='\t')

# batct_plif()
import statistics

def analysis():
    df = pd.read_csv(parent_dir+'/plif.txt', sep='\t')
    for taste in [0,1]:
        df1 = df[df['umami']==taste]
        # with open(parent_dir+f'/{taste}_mean.txt', 'w') as f:
        #     f.write('\t'.join(['T1R1', 'T1R3', 'mGluR1', 'mGluR4', 'T2R46', 'T2R1', 'T2R14', 'T2R4'])+'\n')
        #     for lenth in range(1,16):
        #         df2 = df1[df1['lenth']==lenth]
        #         ls = []
        #         for rec in ['T1R1', 'T1R3', 'mGluR1', 'mGluR4', 'T2R46', 'T2R1', 'T2R14', 'T2R4']:
        #             try:
        #                 mean = statistics.mean(list(df2[rec]))
        #                 ls.append(str(mean))
        #             except:
        #                 ls.append(str(0))
        #         f.write('\t'.join(ls)+'\n')
        # with open(parent_dir+f'/{taste}_std.txt', 'w') as f:
        #     f.write('\t'.join(['T1R1', 'T1R3', 'mGluR1', 'mGluR4', 'T2R46', 'T2R1', 'T2R14', 'T2R4'])+'\n')
        #     for lenth in range(1,16):
        #         df2 = df1[df1['lenth']==lenth]
        #         ls = []
        #         for rec in ['T1R1', 'T1R3', 'mGluR1', 'mGluR4', 'T2R46', 'T2R1', 'T2R14', 'T2R4']:
        #             try:
        #                 std = statistics.stdev(list(df2[rec]))
        #                 ls.append(str(std))
        #             except:
        #                 ls.append(str(0))
        #         f.write('\t'.join(ls)+'\n')
analysis()