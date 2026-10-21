if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_dir}")

    rng = random.Random(seed)
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_val = int(len(files) * fraction)
        for i, src in enumerate(files):
            dest = (val_out if i < n_val else train_out) / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)
    return train_out, val_out


@torch.inference_mode()
def evaluate_local(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        output = model(images.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(output.argmax(dim=1).cpu().tolist())
    return metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Training EfficientNet-B0 Preview 2")
    parser.add_argument("--data", default="./train", help="Folder dataset resmi train/")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/preview2/efficientnet_b0_best.pth")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs dan batch-size harus positif")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data).expanduser().resolve()
    train_dir, val_dir = prepare_split(data_dir, args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    if train_ds.classes != CLASSES or val_ds.classes != CLASSES:
        raise RuntimeError("Urutan kelas tidak sesuai kontrak evaluate.py resmi")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    print(f"Device: {device} | Train: {len(train_ds)} | Validasi: {len(val_ds)}")
    print("Training EfficientNet-B0 pretrained ImageNet V1")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)

        scheduler.step()
        result = evaluate_local(model, val_loader, device)
        f1, acc = result["macro_f1"], result["accuracy"]
        print(
            f"Epoch {epoch}/{args.epochs} | Loss: {loss_sum / len(train_ds):.4f} "
            f"| Macro-F1: {f1:.4f} | Acc: {acc:.4f} "
            f"| Kertas: {result['per_class_f1']['kertas_kardus']:.4f}"
        )
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), output_path)
            print("  -> Checkpoint terbaik eksperimen disimpan")

    print(f"Macro-F1 terbaik: {best_f1:.4f} | Epoch terbaik: {best_epoch}")
    print(f"Checkpoint: {output_path}")
    print("submission/model.pth tidak diubah.")


if __name__ == "__main__":
    main()
