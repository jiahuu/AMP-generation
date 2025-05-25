import torch
from transformers import EsmTokenizer, EsmModel
from torch.nn.utils.rnn import pad_sequence
from model import MCLAMPClassifier


def extract_esm2_embeddings(sequences, model_name="facebook/esm2_t33_650M_UR50D"):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    esm_model.eval()
    batch = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = esm_model(**batch)
    embeddings = outputs.last_hidden_state
    input_mask = batch["attention_mask"]
    valid_embeddings = [embeddings[i, 1:input_mask[i].sum().item()-1] for i in range(len(sequences))]
    return pad_sequence(valid_embeddings, batch_first=True)


def predict(model, sequences, device="cpu"):
    model.eval()
    embeddings = extract_esm2_embeddings(sequences).to(device)
    with torch.no_grad():
        preds, weights = model(embeddings)
    return preds.cpu(), weights.cpu()


if __name__ == "__main__":
    sequences = ["GWLNKKIKKAWRKFHEIFSK", "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MCLAMPClassifier(embedding_dim=1280, hidden_dim=256).to(device)
    model.load_state_dict(torch.load("mcl_amp_model.pt", map_location=device))

    predictions, confidences = predict(model, sequences, device)
    for seq, pred, conf in zip(sequences, predictions, confidences):
        print(f"Sequence: {seq}\nPrediction: {pred:.4f}\nExpert Confidences: {conf.tolist()}\n")
