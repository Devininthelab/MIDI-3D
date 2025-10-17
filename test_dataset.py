from dataset import SyntheticDataset
import torch
train_dataset = SyntheticDataset(
        root_dir='./synthetic_data',
        category='category',
        num_models_per_image=4,
    )

print(f"Number of training samples: {len(train_dataset)}")
print(f"Sample 0: {train_dataset[0].keys()}")


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2
    )

for i, batch in enumerate(train_loader):
    if i >= 8:
        break
    print(f"Batch {i}:")
    print(batch)