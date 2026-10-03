"""
EP1 - Processamento de Linguagem Natural

Modelo final:
BERTimbau (neuralmind/bert-base-portuguese-cased)

Configuração selecionada após comparação com o baseline:
- MAX_LENGTH: 256
- BATCH_SIZE: 8
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Warmup: 10%
- Épocas: 3
"""

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TRAIN_PATH = "train.xlsx"
TEST_PATH = "test1.xlsx"
OUTPUT_PATH = "test1_classificado.xlsx"

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

label2id = {
    "c1": 0,
    "c234": 1,
    "c5": 2
}

id2label = {
    0: "c1",
    1: "c234",
    2: "c5"
}


# ============================================================
# DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Dispositivo utilizado:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

print("\nCarregando dados...")

train = pd.read_excel(TRAIN_PATH)
test = pd.read_excel(TEST_PATH)

X_train = train["resp_text"].fillna("").astype(str)
y_train = train["clarity"]

X_test = test["resp_text"].fillna("").astype(str)

print("Registros de treinamento:", len(train))
print("Registros de teste:", len(test))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# DATASET DE TREINAMENTO
# ============================================================

class TrainDataset(Dataset):

    def __init__(self, textos, labels):

        self.textos = textos.tolist()

        self.labels = [
            label2id[label]
            for label in labels.tolist()
        ]

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.textos[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }


# ============================================================
# DATASET DE TESTE
# ============================================================

class TestDataset(Dataset):

    def __init__(self, textos):

        self.textos = textos.tolist()

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.textos[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0)
        }


# ============================================================
# DATALOADER DE TREINAMENTO
# ============================================================

train_dataset = TrainDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nTotal usado no treinamento:", len(train_dataset))
print("Batches por época:", len(train_loader))


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

print("\nCarregando BERTimbau...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    label2id=label2id,
    id2label=id2label
)

model = model.to(device)


# ============================================================
# OTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = (
    len(train_loader) * EPOCHS
)

warmup_steps = int(
    total_steps * 0.10
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


# ============================================================
# TREINAMENTO FINAL
# ============================================================

print("\nIniciando treinamento final...")

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"ÉPOCA {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    model.train()

    total_loss = 0

    barra = tqdm(
        train_loader,
        desc="Treinando"
    )

    for batch in barra:

        optimizer.zero_grad()

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        labels = (
            batch["labels"]
            .to(device)
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        barra.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    media_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Loss médio: {media_loss:.4f}"
    )


# ============================================================
# TESTE REAL
# ============================================================

print("\nTreinamento concluído.")
print("Iniciando classificação do test1.xlsx...")


test_dataset = TestDataset(X_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model.eval()

previsoes = []

with torch.no_grad():

    for batch in tqdm(
        test_loader,
        desc="Classificando"
    ):

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        previsoes.extend(
            preds.cpu().numpy()
        )


# ============================================================
# CONVERSÃO DAS PREVISÕES
# ============================================================

previsoes_labels = [
    id2label[int(pred)]
    for pred in previsoes
]

test["clarity"] = previsoes_labels


# ============================================================
# VALIDAÇÕES
# ============================================================

print("\nQuantidade de previsões:", len(previsoes_labels))

print(
    "Valores vazios:",
    test["clarity"].isna().sum()
)

print("\nDistribuição das previsões:")

print(
    test["clarity"].value_counts()
)


# ============================================================
# EXPORTAÇÃO
# ============================================================

test.to_excel(
    OUTPUT_PATH,
    index=False
)

print("\n==========================================")
print("PROCESSO FINALIZADO")
print("==========================================")

print(
    f"Arquivo criado: {OUTPUT_PATH}"
)
