# =========== PRÉ-PROCESSAMENTO DOS DADOS ==========
# Leitura dos dados
import tensorflow_datasets as tfds

# Manipulação de dados
import pandas as pd

# Matemática
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

# Química
from rdkit.Chem import GetPeriodicTable
from rdkit import Chem
from rdkit.Chem import AllChem

# Automatização
import subprocess
from pathlib import Path

print('Importação das bibliotecas: CONCLUÍDO COM SUCESSO')

# LEITURA DOS DADOS
ds = tfds.load('qm9', split='train', shuffle_files=False)

print('Leitura dos dados: CONCLUÍDO COM SUCESSO')

# FEATURIZAÇÃO
try:
    PT = GetPeriodicTable()
    def atomic_number(symbol):
        return PT.GetAtomicNumber(symbol)
except Exception:
    # Mapeamentos dos números atômicos
    ATOMIC_NUM = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9}
    def atomic_number(symbol):
        return ATOMIC_NUM[symbol]

# ---------- 1) Construtor da Matriz de Coulomb ----------
def coulomb_matrix(Z, coords, max_atoms=None, diag_exp=2.4, eps=1e-8):
    """
    Z: 1D array-like de números atômicos, shape (n_atoms,)
    coords: array-like shape (n_atoms, 3)
    max_atoms: int, tamanho fixo (pad com zeros se n_atoms < max_atoms)
    diag_exp: expoente para diagonal (0.5 * Z_i**diag_exp)
    eps: pequena constante para evitar divisão por zero (distância 0)
    Returns: Coulomb matrix padded to (max_atoms, max_atoms)
    """
    Z = np.array(Z, dtype=float)
    coords = np.array(coords, dtype=float)
    n = len(Z)
    if max_atoms is None:
        max_atoms = n
    # Matriz de distâncias (n x n)
    dists = cdist(coords, coords)  # Eficiente em C
    # Inicia a matriz
    C = np.zeros((max_atoms, max_atoms), dtype=float)
    # Preencher bloco (n x n)
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, i] = 0.5 * (Z[i] ** diag_exp)
            else:
                dij = dists[i, j]
                C[i, j] = (Z[i] * Z[j]) / (dij + eps)
    return C

# ---------- 2) Ordenamento / Invariância ----------
def sort_coulomb_matrix(C):
    """
    Ordena linhas/colunas da matriz C por norma L2 decrescente das linhas.
    Retorna a matriz ordenada.
    """
    # Norma por linha
    row_norms = np.linalg.norm(C, axis=1)
    # Organiza índices (descer)
    order = np.argsort(-row_norms)
    C_sorted = C[order][:, order]
    return C_sorted, order

# ---------- 3) Flattening / Trigângulo ----------
def flatten_upper_triangle(C):
    """Retorna vetor com elementos da metade superior (incluindo diagonal)."""
    idx = np.triu_indices_from(C)
    return C[idx]

def flatten_full(C):
    """Flatten completo (row-major)."""
    return C.flatten()

# ---------- 4) Espectro de autovalores ----------
def coulomb_eigenvalues(C, k=None):
    """
    Retorna autovalores (ordenados decrescentemente) do bloco ativo de C.
    k: número de primeiros autovalores a retornar; se None, retorna todos.
    """
    # C pode ter padding zeros; para estabilidade usamos eigh (symmetric)
    evals = eigh(C, eigvals_only=True)
    # Ordem decrescente por valor absoluto (ou valor real)
    order = np.argsort(-np.abs(evals))
    evals_sorted = evals[order]
    if k is not None:
        # pad se necessário
        res = np.zeros(k, dtype=float)
        n = min(len(evals_sorted), k)
        res[:n] = evals_sorted[:n]
        return res
    else:
        return evals_sorted

# ---------- 5) Pipeline completo para uma molécula ----------
def featurize_one_molecule(atom_symbols, coords, max_atoms=29,
                           use='sorted_flatten', diag_exp=2.4):
    """
    atom_symbols: list of chemical symbols e.g. ['C','H','H','O',...]
    coords: numpy array (n_atoms, 3)
    use: 'sorted_flatten' | 'spectrum' | 'flatten_full' | 'upper_triangle'
    Returns: 1D numpy array feature vector
    """
    Z = [atomic_number(s) for s in atom_symbols]
    C = coulomb_matrix(Z, coords, max_atoms=max_atoms, diag_exp=diag_exp)
    if use == 'sorted_flatten':
        Csorted, _ = sort_coulomb_matrix(C)
        feat = flatten_upper_triangle(Csorted)  # Usa-se triangulação para reduzir dimensionalidade
    elif use == 'flatten_full':
        Csorted, _ = sort_coulomb_matrix(C)
        feat = flatten_full(Csorted)
    elif use == 'upper_triangle':
        feat = flatten_upper_triangle(C)
    elif use == 'spectrum':
        feat = coulomb_eigenvalues(C, k=max_atoms)  # length = max_atoms
    else:
        raise ValueError("use deve ser 'sorted_flatten'|'spectrum'|'flatten_full'|'upper_triangle'")
    # Garante que seja finito
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat

# MATRIZ DE COULOMB
def coulomb_matrix(atoms, positions, max_atoms=29):
    """
    Gera uma matriz de Coulomb padronizada (com preenchimento) para uma molécula.

    Parameters
    ----------
    atoms : array-like
        Lista de números atômicos (ex: [6, 1, 1, 8])
    positions : array-like
        Coordenadas 3D correspondentes (shape: [n_atoms, 3])
    max_atoms : int
        Tamanho máximo da molécula no dataset (QM9 tem até 29 átomos)

    Returns
    -------
    M : np.ndarray
        Matriz de Coulomb quadrada de shape (max_atoms, max_atoms)
    """
    n = len(atoms)
    Z = np.array(atoms, dtype=float)
    R = np.array(positions, dtype=float)

    # Cria matriz de distâncias |R_i - R_j|
    dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    np.fill_diagonal(dist, 1.0)  # Evita divisão por zero

    # Termos fora da diagonal
    M = np.outer(Z, Z) / dist
    # Termos diagonais
    np.fill_diagonal(M, 0.5 * Z ** 2.4)

    # Padroniza para tamanho fixo
    M_padded = np.zeros((max_atoms, max_atoms))
    M_padded[:n, :n] = M

    return M_padded

# ITERAÇÃO SOBRE O QM9
# Carrega o dataset (já cacheado)
ds = tfds.load('qm9', split='train', shuffle_files=False)
ds = tfds.as_numpy(ds)  # Converte de tensores do TensorFflow para arrays NumPy

X, y = [], []

for sample in ds:
    atoms = sample['charges'][:sample['num_atoms']]
    positions = sample['positions'][:sample['num_atoms']]
    target = sample['U0']

    M = coulomb_matrix(atoms, positions)
    X.append(M.flatten())  # Flattening para vetor
    y.append(target)

X = np.array(X)
y = np.array(y)

# CONVERSÃO PARA DATAFRAME
df = pd.DataFrame(X, columns=[f'coulomb_{i}' for i in range(X.shape[1])])
df['target'] = y  # Adiciona a coluna do target

features = df.columns

for f in features:
    if (df[f] == 0.0).all():
        df.drop(f, axis=1, inplace=True)

print('Processamento dos dados: CONCLUÍDO COM SUCESSO')


# ========== AUTOMATIZAÇÃO QM9 -> XTB ==========
# 1.
smiles_list = []

for i, exemplo in enumerate(ds):
    # Decodifica o SMILES
    smiles = exemplo['SMILES'].decode('utf-8')  # bytes → str
    smiles_list.append(smiles)

print('Decodificação do SMILES: CONCLUÍDO COM SUCESSO')

# 2.
def smiles_to_xyz(smiles, i, output_path="/home/mateus25032/work/Projeto_Final_ML/rdkit_molecules"):
    """
    Gera um arquivo `.xyz` a partir de um SMILES válido, com otimização geométrica via RDKit.

    # Parâmetros
    smiles (str): SMILES da molécula.
    i (int): Índice da molécula no dataset.
    output_path (path, str): Endereço do diretório onde as geometrias serão salvas em arquivos `.xyz`.

    # Retorna
       Retorna `True` se a molécula foi salva com sucesso, `False` caso contrário.
    """
    # Validação do SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[ERRO] SMILES inválido: {smiles}")
        return False

    # Adiciona hidrogênios
    mol = Chem.AddHs(mol)

    # Tenta gerar a geometria 3D com ETKDGv3 (método mais moderno e estável)
    params = AllChem.ETKDGv3()
    params.randomSeed = 88
    res = AllChem.EmbedMolecule(mol, params)

    # Se falhar, tenta novamente com coordenadas aleatórias
    if res != 0:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
            return False

    # Otimização geométrica com UFF
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except ValueError:
        return False

    # Salva o arquivo .xyz
    filename = f"{output_path}/molecule_{i}.xyz"
    with open(filename, "w") as f:
        f.write(Chem.MolToXYZBlock(mol))

    return True

for i, smiles in enumerate(smiles_list):
    try:
        smiles_to_xyz(smiles, i, output_path="/home/mateus25032/work/Projeto_Final_ML/rdkit_molecules")
    except ValueError:
        continue

print('Processamento das geometrias (rdkit): CONCLUÍDO COM SUCESSO')

# 3.
xtb_path = Path(r'/home/mateus25032/work/Projeto_Final_ML/xtb-linux/bin/xtb') # Diretório do xtb
xyz_dir = Path(r'/home/mateus25032/work/Projeto_Final_ML/rdkit_molecules')  # Diretório com as geometrias (arquivos .xyz)
output_dir = Path('xtb_results')
output_dir.mkdir(exist_ok=True)
gfn_model = 2

def run_xtb(xyz_path, command_line):
    out_path = output_dir / f'{xyz_path.stem}.out'
    with open(out_path, 'w') as f:
        subprocess.run(
            command_line,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=1200
        )
    return out_path

for xyz_file in xyz_dir.glob('*.xyz'):
    out_file = run_xtb(xyz_file, [str(xtb_path), str(xyz_file), "--opt", f"--gfn", str(gfn_model)])

print('Otimização das geometrias (xTB): CONCLUÍDO COM SUCESSO')

for xyz_file in xyz_dir.glob('*.xyz'):
    out_file = run_xtb(xyz_file, [str(xtb_path), str(xyz_file), '--hess'])

print('Cálculo do Hessiano (xTB): CONCLUÍDO COM SUCESSO')

# 4.
path = Path(r'/home/mateus25032/work/Projeto_Final_ML/xtb_results')

def parser(prop, out_path):
    """
    Leitura e extração dos dados de propriedades químicas presentes nos arquivos `.out` gerados pelo xTB.
    
    # Parâmetros
    - prop (str): Propriedade química a ser extraída. Pode assumir os valores: "dipole"

    - out_path (str): Endereço do arquivo .out a ser lido.
    
    # Retorna
    Valor numérico da propriedade química escolhida (float).
    """
    
    match prop:
        case ('dipole'):
            with open(out_path, 'r', encoding='latin-1') as f:
                c = 0
                for line in f:
                    if 'molecular dipole' in line:
                        c += 1
                    if 'full' in line and c == 1:
                        return float(line.split()[-1])
        case 'HOMO':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if '(HOMO)' in line:
                        return float(line.split()[-2])
        case 'LUMO':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if '(LUMO)' in line:
                        return float(line.split()[-2])
        case 'ZPE':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if 'zero point energy' in line:
                        return float(line.split()[-3])
        case 'H':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if 'TOTAL ENTHALPY' in line:
                        return float(line.split()[-3])
        case 'U0':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if 'TOTAL ENERGY' in line:
                        return float(line.split()[3])
        case 'G':
            with open(out_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if 'TOTAL FREE ENERGY' in line:
                        return float(line.split()[-3])
                    
dipole_list = []

for file in path.glob('*.out'):
    dipole = parser('dipole', file)
    dipole_list.append(dipole)

homo_list = []

for file in path.glob('*.out'):
    E_HOMO = parser('HOMO', file)
    homo_list.append(E_HOMO)

lumo_list = []

for file in path.glob('*.out'):
    E_LUMO = parser('LUMO', file)
    lumo_list.append(E_LUMO)

gap_list = np.array(homo_list) - np.array(lumo_list)

zpe_list = []

for file in path.glob('*.out'):
    ZPE = parser('ZPE', file)
    zpe_list.append(ZPE)

enthalpy_list = []

for file in path.glob('*.out'):
    H = parser('H', file)
    enthalpy_list.append(H)

R = 8.31446261815324  # J/mol·K
T = 298.15
Eh_to_Jmol = 2625.499638  # 1 Eh = 2625.5 kJ/mol
RT_Eh = (R * T / 1000) / Eh_to_Jmol  # converte R*T para Hartree

U = np.array(enthalpy_list) - RT_Eh

U0_xtb_list = []

for file in path.glob('*.out'):
    U0 = parser('U0', file)
    U0_xtb_list.append(U0)

gibbs_list = []

for file in path.glob('*.out'):
    G = parser('G', file)
    gibbs_list.append(G)

U0_qm9__list = []

# Extrai U0 do QM9
for i, exemplo in enumerate(ds):
    u0 = exemplo['U0']
    U0_qm9__list.append(u0)

delta_qm9_xtb = np.array(U0_qm9__list[:1]) - np.array(U0_xtb_list)

print('Extração das propriedades: CONCLUÍDO COM SUCESSO')

# ========== DATASET XTB ==========
xtb_dataset = pd.DataFrame({
    'Dipole': dipole_list,
    'E_HOMO': homo_list,
    'E_LUMO': lumo_list,
    'gap_HOMO-LUMO': gap_list,
    'ZPE': zpe_list,
    'H': enthalpy_list,
    'U': U,
    'U0': U0_xtb_list,
    'G': gibbs_list,
    'Delta': delta_qm9_xtb
})

xtb_dataset.to_csv('xtb_dataset.csv')
print('Dataset processado e salvo: CONCLUÍDO COM SUCESSO')

print('########## PROCESSAMENTO DOS DADOS CONCLUÍDO COM SUCESSO ###########')
