import argparse
import os
import torch
from monai.data import decollate_batch, DataLoader, Dataset
from monai.metrics import ROCAUCMetric
from monai.networks.nets import ResNet
from monai.transforms import (
    Compose,
    AsDiscrete,
    LoadImaged,
    AddChanneld,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    ToTensord,
    Orientationd,
    Spacingd,
)
from monai.utils import set_determinism
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="MONAI Deep Learning Pipeline for AI-Assisted Medical Imaging Robustness Study.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save trained models and predictions.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model for inference.")
    parser.add_argument("--inference_input", type=str, default=None, help="Path to an image for single image inference.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Pipeline mode: train or inference.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_determinism(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define Transforms
    # Estas transformações são exemplos e devem ser ajustadas ao seu dataset específico
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, 
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=(224, 224)), # Ajustar para o tamanho de entrada da ResNet
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, 
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=(224, 224)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # 2. Prepare Data (Placeholder)
    # Você precisará adaptar esta parte para carregar seus dados reais.
    # Exemplo: Criar uma lista de dicionários com caminhos para imagens e labels.
    # Por exemplo, para um dataset de classificação binária:
    # train_files = [{"image": "path/to/image1.png", "label": 0}, ...]
    # val_files = [{"image": "path/to/image_val1.png", "label": 1}, ...]
    
    # Para fins de demonstração, vamos criar dados dummy
    print("Criando dados dummy para demonstração. Por favor, substitua pelos seus dados reais.")
    dummy_data = []
    for i in range(100):
        dummy_data.append({"image": os.path.join(args.data_dir, f"image_{i}.png"), "label": i % 2})
    
    train_files = dummy_data[:80]
    val_files = dummy_data[80:]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers pode ser ajustado

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # 3. Define Network, Optimizer, and Loss Function
    # Usando ResNet-101 como mencionado no artigo, com 1 saída para classificação binária
    model = ResNet(block=ResNet.ResNet50Layer, layers=[3, 4, 23, 3], block_inplanes=[64, 128, 256, 512], 
                   feed_forward=True, num_classes=1).to(device) # num_classes=1 para classificação binária com sigmoid

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    roi_auc_metric = ROCAUCMetric()

    # 4. Training and Validation Loop
    if args.mode == "train":
        best_metric = -1
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            model.train()
            epoch_loss = 0
            for batch_data in tqdm(train_loader, desc="Training"):
                inputs, labels = batch_data["image"].to(device), batch_data["label"].float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Training Loss: {epoch_loss / len(train_loader):.4f}")

            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y_true = torch.tensor([], dtype=torch.long, device=device)
                for val_data in tqdm(val_loader, desc="Validation"): 
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_images)
                    y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y_true = torch.cat([y_true, val_labels], dim=0)
                
                # AsDiscrete para converter logits em probabilidades e depois em classes
                y_pred_act = torch.sigmoid(y_pred)
                y_pred_binary = (y_pred_act >= 0.5).int()

                # ROCAUCMetric espera probabilidades e labels
                roi_auc_metric(y_pred_act, y_true.unsqueeze(1))
                auc_result = roi_auc_metric.aggregate().item()
                roi_auc_metric.reset()

                print(f"Validation AUC: {auc_result:.4f}")

                if auc_result > best_metric:
                    best_metric = auc_result
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_metric_model.pth"))
                    print("Saved new best metric model!")

    # 5. Inference
    if args.mode == "inference":
        if args.model_path is None:
            print("Erro: Caminho para o modelo pré-treinado (--model_path) é necessário para inferência.")
            return
        if args.inference_input is None:
            print("Erro: Caminho para a imagem de entrada (--inference_input) é necessário para inferência.")
            return

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        # Transformações para inferência (sem aumento de dados)
        inference_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, 
                                     b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
                Resized(keys=["image"], spatial_size=(224, 224)),
                ToTensord(keys=["image"]),
            ]
        )

        # Carregar e pré-processar a imagem de inferência
        inference_data = inference_transforms({"image": args.inference_input})
        input_image = inference_data["image"].unsqueeze(0).to(device) # Adiciona dimensão de batch

        with torch.no_grad():
            output = model(input_image)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability >= 0.5 else 0
        
        print(f"\n--- Inference Result for {args.inference_input} ---")
        print(f"Predicted Probability: {probability:.4f}")
        print(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
