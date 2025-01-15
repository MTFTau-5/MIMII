import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from feeder import AudioDataset
from net import AM
from util import yaml_parser


def main():
    (
        input_dim, num_classes, num_heads, num_layers, dim_feedforward,
        batch_size, pkl_file_path, num_devices, test_size, num_epochs, random_state, lr
    ) = yaml_parser()

    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    
    audio_dataset = AudioDataset(pkl_file_path, num_devices)


    train_data, test_data = train_test_split(audio_dataset, test_size=test_size, random_state=random_state)
    
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    model = AM(input_dim, num_classes, num_heads, num_layers, dim_feedforward)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for batch_features, batch_device_nums, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")





    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_device_nums, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()


    print(f"Final test loss: {test_loss / len(test_loader)}")
    print(f"Final test accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    main()
