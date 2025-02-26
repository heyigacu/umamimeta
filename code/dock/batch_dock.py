import pandas as pd
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
ADCP = '/home/hy/Softwares/ADFRsuite/ADFRsuite-1.1dev/bin/adcp'
VINA = '/home/hy/Softwares/vina/autodock_vina_1_1_2_linux_x86/bin/vina'


def batct_dock(rec_input_path=parent_dir+'/ummrec.csv', pep_input_path=parent_dir+'/ummpepYEED.csv', 
               trg_dir=parent_dir+'/receptor/trg', dock_dir=parent_dir+'/dock', 
               rec_pdbqt_dir=parent_dir+'/receptor/pdbqt', pep_pdbqt_dir=parent_dir+'/peptide/pdbqt',):
    df_rec = pd.read_csv(rec_input_path, sep='\t', header=0)
    df_pep = pd.read_csv(pep_input_path, sep='\t', header=0)

    for index_rec, row_rec in df_rec.iterrows():
        trg_path = f"{trg_dir}/{row_rec['pocket']}.trg"
        trg_name = f"{row_rec['pocket']}.trg"
        receptor_pdbqt_path = f"{rec_pdbqt_dir}/{row_rec['receptor']}.pdbqt"
        for index_pep, row_pep in df_pep.iterrows():
            project_dir = dock_dir+f"/{row_rec['pocket']}/{row_pep['Sequence']}/"
            project_name = f"{row_rec['pocket']}_{row_pep['Sequence']}"
            pep_pdbqt_path = f"{pep_pdbqt_dir}/{row_pep['Sequence']}.pdbqt"
            if 16 > len(row_pep['Sequence']) >= 5: 
                if not os.path.exists(project_dir): os.makedirs(project_dir)
                pep_seq = row_pep['Sequence'].lower()
                dock_out_log = f"{project_dir}/{project_name}.log"
                os.system(f"cp {trg_path} {project_dir}/")
                os.system(f'cd {project_dir}; {ADCP} -t {trg_name} -s {pep_seq} -N 10 -n 500000 -o {project_name} > {dock_out_log} ')
            if len(row_pep['Sequence']) < 5:
                if not os.path.exists(project_dir): os.makedirs(project_dir)
                dock_out_path = f"{project_dir}/{project_name}_out.pdbqt"
                dock_out_log = f"{project_dir}/{project_name}.log"
                os.system(f"cd {project_dir}; {VINA} --center_x {row_rec['center_x']} --center_y {row_rec['center_y']} --center_z {row_rec['center_z']} \
                          --size_x {row_rec['size_x']} --size_y {row_rec['size_y']} --size_z {row_rec['size_z']} \
                          --receptor {receptor_pdbqt_path} --ligand {pep_pdbqt_path} --out {dock_out_path} --log {dock_out_log} --num_modes 10 --exhaustiveness 8")
            else:
                pass

"""

adcp -t T1R1_big.trg -s  gkkrseapphif -N 10 -n 100000 -o T1R1_GKKRSEAPPHIF > T1R1_GKKRSEAPPHIF.log
adcp -t T1R3_big.trg -s  vtadesqqdvlk -N 10 -n 100000 -o T1R3_VTADESQQDVLK > T1R3_VTADESQQDVLK.log



cd /home/hy/Documents/protein-peptide/umamimeta/dock/dock/T1R3/NFNNQLDQQTPR
cp ~/Documents/protein-peptide/umamimeta/dock/receptor/pdb/T1R3_big.trg .
adcp -t T1R3_big.trg -s  nfnnqldqqtpr -N 10 -n 100000 -o T1R3_NFNNQLDQQTPR > T1R3_NFNNQLDQQTPR.log

cd /home/hy/Documents/protein-peptide/umamimeta/dock/dock/T1R3/AVLEEAQKVELK
cp ~/Documents/protein-peptide/umamimeta/dock/receptor/pdb/T1R3_big.trg .
adcp -t T1R3_big.trg -s  avleeaqkvelk -N 10 -n 100000 -o T1R3_AVLEEAQKVELK > T1R3_AVLEEAQKVELK.log





mkdir /home/hy/Documents/protein-peptide/umamimeta/dock/dock/T2R4/FFVAPFPEVFGK
cd /home/hy/Documents/protein-peptide/umamimeta/dock/dock/T2R4/FFVAPFPEVFGK
cp ~/Documents/protein-peptide/umamimeta/dock/receptor/trg/T2R4_big.trg .
/home/hy/Softwares/ADFRsuite/ADFRsuite-1.1dev/bin/adcp -t T2R4_big.trg -s ffvapfpevfgk -N 10 -n 100000 -o T2R4_FFVAPFPEVFGK > T2R4_FFVAPFPEVFGK.log




"""

batct_dock()


