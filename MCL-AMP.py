import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmTokenizer, EsmModel
from torch.nn.utils.rnn import pad_sequence


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


def extract_esm2_embeddings(sequences, model_name="facebook/esm2_t33_650M_UR50D"):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    esm_model.eval()

    batch = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = esm_model(**batch)
        embeddings = outputs.last_hidden_state

    input_mask = batch["attention_mask"]
    valid_embeddings = []
    for i in range(len(sequences)):
        seq_len = input_mask[i].sum().item()
        valid_embeddings.append(embeddings[i, 1:seq_len-1])  # remove CLS/EOS

    return pad_sequence(valid_embeddings, batch_first=True)  # [B, max_len, 1280]


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



if __name__ == "__main__":
    model = MCLAMPClassifier(embedding_dim=1280, hidden_dim=256).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Dummy example for illustration
    sequences = ["GWLNKKIKKAWRKFHEIFSK", "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"]
    labels = torch.tensor([1, 0])
    from torch.utils.data import DataLoader, TensorDataset

    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    dataset = SequenceDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for epoch in range(3):
        avg_loss = train_model(model, dataloader, optimizer, device="cuda")
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    preds = predict(model, sequences, device="cuda")
    print("Predictions:", preds)

