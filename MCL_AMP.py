import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from transformers import EsmTokenizer, EsmModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPExpert, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.mean(dim=1)
        feat = self.encoder(x)
        out = self.classifier(feat)
        return torch.sigmoid(out).squeeze(), feat


class CNNExpert(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(CNNExpert, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).squeeze(-1)
        feat = self.fc(x)
        out = self.classifier(feat)
        return torch.sigmoid(out).squeeze(), feat


class BiLSTMExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMExpert, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.fc(out[:, -1, :])
        out = self.classifier(feat)
        return torch.sigmoid(out).squeeze(), feat


class VotingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(VotingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        return self.fc(features)


class MCLAMPClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MCLAMPClassifier, self).__init__()
        self.mlp_expert = MLPExpert(embedding_dim, hidden_dim)
        self.cnn_expert = CNNExpert(embedding_dim, hidden_dim)
        self.lstm_expert = BiLSTMExpert(embedding_dim, hidden_dim)
        self.voting_net = VotingNetwork(hidden_dim * 3, num_experts=3)

    def forward(self, x):
        mlp_out, mlp_feat = self.mlp_expert(x)
        cnn_out, cnn_feat = self.cnn_expert(x)
        lstm_out, lstm_feat = self.lstm_expert(x)

        concat_feat = torch.cat([mlp_feat, cnn_feat, lstm_feat], dim=1)  # [B, hidden_dim * 3]
        weights = self.voting_net(concat_feat)  # [B, 3]

        outputs = torch.stack([mlp_out, cnn_out, lstm_out], dim=1)  # [B, 3]
        weighted_output = (outputs * weights).sum(dim=1)  # [B]

        return weighted_output, weights

# 其余部分（loss、提取embedding、训练等）无需修改，使用 weighted_output 即是最终输出


def confident_hinge_loss(preds, labels, weights, alpha=1.0, beta=0.05):
    bce_loss = F.binary_cross_entropy(preds, labels.float(), reduction='none')
    oracle_loss = (weights * bce_loss.unsqueeze(1)).sum(dim=1).mean()
    confidence_penalty = 0
    probs = preds.detach()
    for i in range(len(labels)):
        y_true = labels[i].long().item()
        for c in [0, 1]:
            if c != y_true:
                diff = probs[i] if c == 1 else (1 - probs[i])
                margin = torch.clamp(diff - (1 - diff) + beta, min=0)
                confidence_penalty += margin
    confidence_penalty = confidence_penalty / len(labels)
    return oracle_loss + alpha * confidence_penalty

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def extract_esm2_embeddings(sequences, model_name="facebook/esm2_t33_650M_UR50D", max_len=30):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    esm_model.eval()

    batch = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        outputs = esm_model(**batch)
        embeddings = outputs.last_hidden_state

    input_mask = batch["attention_mask"]
    valid_embeddings = []
    for i in range(len(sequences)):
        seq_len = input_mask[i].sum().item()
        valid_embed = embeddings[i, 1:seq_len-1]  # remove CLS/EOS
        pad_len = max_len - valid_embed.size(0)
        pad_tensor = torch.zeros(pad_len, valid_embed.size(1))  # [pad_len, 1280]
        valid_embed = torch.cat([valid_embed, pad_tensor], dim=0)

        valid_embeddings.append(valid_embed)

    return torch.stack(valid_embeddings)


def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for sequences, labels in dataloader:
        optimizer.zero_grad()
        embeddings = extract_esm2_embeddings(sequences).to(device)
        labels = labels.to(device)
        preds, weights = model(embeddings)
        loss = confident_hinge_loss(preds, labels, weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

epochs = 100
batch_size = 256
learning_rate = 1e-4

def evaluate_metrics(model, dataloader, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            embeddings = extract_esm2_embeddings(sequences).to(device)
            labels = labels.to(device)
            preds, weights = model(embeddings)  # ✅ 只取预测结果
            loss = confident_hinge_loss(preds, labels, weights)  # 与训练一致
            total_loss += loss.item()
            pred_labels = (preds > 0.5).long()  # ✅ 二分类，0/1 预测
            all_preds.extend(pred_labels.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(all_labels, all_preds)

    return total_loss/len(dataloader), acc, sn, sp, mcc


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    sequences = df["Seq"].tolist()
    labels = torch.tensor(df["Label"].tolist())

    dataset = SequenceDataset(sequences, labels)
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = MCLAMPClassifier(embedding_dim=1280, hidden_dim=256).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = 1.0

    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, device="cuda")
        train_losses.append(train_loss)
        val_loss, _, _, _, _ = evaluate_metrics(model, val_loader, device="cuda")
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, train_Loss: {train_loss:.4f}, val_Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    model.load_state_dict(torch.load("best_model.pt"))
    _, test_acc, test_sn, test_sp, test_mcc = evaluate_metrics(model, test_loader, device="cuda")
    print(f"Test Acc: {test_acc:.4f}, Sn: {test_sn:.4f}, Sp: {test_sp:.4f}, MCC: {test_mcc:.4f}")
