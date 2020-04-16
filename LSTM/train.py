def train_model(input_size, hidden_size, train_data, lr = 0.01, batch_size=100, epochs=10):
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    model = LSTM(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'[Epoch {epochs}/]  loss: {loss.item(): .4f}')