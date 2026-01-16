import torch
from tqdm import tqdm


def evaluate_model_performance(model, dataloader, device, num_classes, restoration_hook=None):
    model.eval()

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if restoration_hook is not None:
                restoration_hook.set_input(inputs, labels)

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(logits, dim=-1)

            correct = (predictions == labels)
            total_correct += correct.sum().item()
            total_samples += labels.size(0)

            for class_id in range(num_classes):
                class_mask = (labels == class_id)
                if class_mask.sum() > 0:
                    class_predictions = predictions[class_mask]
                    class_labels = labels[class_mask]
                    class_correct[class_id] += (class_predictions == class_labels).sum().item()
                    class_total[class_id] += class_mask.sum().item()

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    class_accs = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    return overall_acc, class_accs
