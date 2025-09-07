# myutility/gnn_utils.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import numpy as np


# --- 1. SMILES를 그래프 데이터로 변환하는 기능 ---

def get_atom_features(atom):
    """원자(atom) 객체로부터 특징 벡터를 생성합니다."""
    # 원자 종류(Symbol), 원자가전자 수(Valence), 형식 전하(Formal Charge) 등
    feature = np.zeros(36)
    # 심볼 (1-18: H-Ar, 19: 기타)
    try:
        atomic_num = atom.GetAtomicNum()
        feature[min(atomic_num, 18) - 1] = 1
    except:
        feature[18] = 1
    # 원자가전자 수
    try:
        valence = atom.GetTotalValence()
        feature[19 + min(valence, 6)] = 1  # 0 to 6
    except:
        feature[19 + 7] = 1  # unknown
    # 형식 전하
    try:
        charge = atom.GetFormalCharge()
        feature[27 + min(charge + 2, 4)] = 1  # -2 to +2
    except:
        feature[27 + 5] = 1  # unknown
    # 방향성(Aromatic) 여부
    feature[33] = atom.GetIsAromatic()
    # 링(Ring)에 속하는지 여부
    feature[34] = atom.IsInRing()
    # 수소 원자 개수
    feature[35] = atom.GetTotalNumHs()
    return torch.tensor(feature, dtype=torch.float)


def smiles_to_graph(smiles: str, y: torch.Tensor):
    """SMILES 문자열과 타겟 값(y)으로부터 PyG Data 객체를 생성합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 원자 특징 추출
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    # 결합 정보 (edge_index) 추출
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)


# --- 2. PyTorch용 커스텀 데이터셋 ---

class MoleculeDataset(Dataset):
    """SMILES 데이터프레임을 PyG 데이터셋으로 변환하는 클래스."""

    def __init__(self, df: pd.DataFrame, target_columns: list):
        super(MoleculeDataset, self).__init__()
        self.smiles = df['SMILES'].tolist()
        self.targets = torch.tensor(df[target_columns].values, dtype=torch.float)

        # 데이터 변환 (SMILES -> Graph)
        self.data_list = [smiles_to_graph(s, self.targets[i]) for i, s in enumerate(self.smiles)]
        # 변환 실패한 데이터(None) 제거
        self.data_list = [d for d in self.data_list if d is not None]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# --- 3. GNN 모델 아키텍처 정의 ---
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNNet(torch.nn.Module):
    """GCN 레이어를 사용한 간단한 GNN 모델."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GCN 레이어 통과
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # 2. Global Pooling (Node-level -> Graph-level)
        x = global_mean_pool(x, batch)

        # 3. 최종 예측을 위한 MLP 통과
        return self.mlp(x)


# --- 4. 훈련 및 평가 함수 ---

def train_epoch(model, loader, optimizer, criterion, device):
    """한 에폭(epoch) 동안 모델을 훈련합니다."""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # data.y를 [batch_size, num_targets] 모양으로 재구성합니다.
        # data.num_graphs가 현재 배치의 실제 크기(batch_size)입니다.
        y_reshaped = data.y.view(data.num_graphs, -1)
        loss = criterion(out, y_reshaped)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    """한 에폭(epoch) 동안 모델 성능을 평가합니다."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            # 훈련 함수와 동일하게 모양을 재구성합니다.
            y_reshaped = data.y.view(data.num_graphs, -1)
            loss = criterion(out, y_reshaped)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)