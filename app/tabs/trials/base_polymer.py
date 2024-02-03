def display(tab):
    tab.write("This tutorial explains how to train a base model for polymers. The fingerprints used here are polyBERT fingerprints.")
    tab.write("You can find the running tutorial on google colab: https://colab.research.google.com/drive/1MqA4rAkh4y_WiZ8VOTpfOpz9YYxX-47i?usp=sharing")
    
    tab.subheader("1. Import necessary libraries")
    code_1 = """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Setting random seed
    random_seed = 123
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    """
    tab.code(code_1, language="python")

    tab.subheader("2. Fetching data from pickle files")
    code_2 = """ 
    # Load DataFrame
    df = pd.read_pickle("updated_polymers.pth")

    # Display DataFrame Head
    df.head()
    """
    tab.code(code_2, language="python")

    tab.subheader("3. Data preparation")
    code_3 = """
     #Setting required variables
    scalar = MinMaxScaler()
    data = df["fingerprint_polyBERT"]
    target = df["Egc"]

    #Spliting the train:test data into 80:20
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=123)

    # Scaling target variable
    target_train = scalar.fit_transform(target_train.values.reshape(-1, 1))
    target_test = scalar.transform(target_test.values.reshape(-1, 1))

    # Creating tensors from data

    #Training Data
    data_train_tensor = torch.tensor(data_train.reset_index(drop = True), dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)

    train_dataset = TensorDataset(data_train_tensor, target_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True)

    #Testing Data

    data_test_tensor = torch.tensor(data_test.reset_index(drop= True), dtype=torch.float32)
    target_test_tensor = torch.tensor(target_test, dtype=torch.float32)

    test_dataset = TensorDataset(data_test_tensor, target_test_tensor)
    test_loader = DataLoader(test_dataset, shuffle= False)
    """
    tab.code(code_3, language = "python")

    tab.subheader("4. Define the neural network")
    code_4 = """
    class MTmodel(nn.Module):
        def __init__(self):
            super(MTmodel, self).__init__()
            self.my_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(600, 992),
                    nn.Dropout(0.101574086),
                    nn.PReLU()
                ),
                nn.Sequential(
                    nn.Linear(992, 1568),
                    nn.Dropout(0.16078567),
                    nn.PReLU()
                ),
                 nn.Linear(1568, 1)
         ])
        
    
        def forward(self, x):
            for layer_step in self.my_layers:
                x = layer_step(x)
            return x
    """
    
    tab.code(code_4, language = "python")
    
    tab.subheader("5. Train your model")
    code_5 = """
    # Initializing the neural network
    net = MTmodel()
    net = net.to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr =  0.000181089)
    EPOCHS = 532
    losses = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
  
        for batch_idx,(data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.view(-1).to(DEVICE)
            optimizer.zero_grad()
            output = net(data)
            loss = F.mse_loss(output.view(-1), target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / batch_idx
        losses.append(epoch_loss)

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss}")
    """
    tab.code(code_5, language = "python")

    tab.subheader("6. Plot the parity plot")
    code_6 = """
    from scipy.stats import gaussian_kde

    # Plot predictions vs. true values
    @torch.no_grad()
    def graphPredictions(model, data_loader , minValue, maxValue):
        # Set the model to inference mode
        model.eval()                               

        # Track predictions
        predictions=[]                             
        # Track the actual labels
        actual=[]  
        loss = []                                
        model = model.to(DEVICE)
    
    

        for (data,target) in (test_loader):
            # Single forward pass
            data, target =  data.to(DEVICE), target.to(DEVICE)
            pred = model(data)                              
        
            # Un-normalize our prediction
            pred = scalar.inverse_transform(pred.cpu().numpy())
            pred = torch.from_numpy(pred).to(DEVICE)
            target_cpu = scalar.inverse_transform(target.cpu().numpy())
            act = torch.from_numpy(target_cpu).to(DEVICE)
            loss = F.mse_loss(pred, act)

            # Save prediction and actual label
            predictions.append(tensor.cpu().item() for tensor in pred)
            actual.append(tensor.cpu().item() for tensor in act)

        pred_list = [item for sublist in predictions for item in sublist]
        act_list = [item for sublist in actual for item in sublist]

    
        mse = mean_squared_error(pred_list, act_list)
        rmse = mean_squared_error(pred_list, act_list, squared = False)
        r2 = r2_score(pred_list, act_list)
        print(f"mse: {mse}, rmse: {rmse}, r2_score: {r2}")

        # Calculate the point density
        xy = np.vstack([pred_list,act_list])
        z = gaussian_kde(xy)(xy)
    
        # Plot actuals vs predictions
        plt.scatter(pred_list, act_list, s= 10, c=z)
        plt.xlabel('Actual Egc')
        plt.ylabel('Predicted Egc')
        plt.plot([minValue,maxValue], [minValue,maxValue])
        plt.xlim(minValue, maxValue)
        plt.ylim(minValue, maxValue)

        # Make the display equal in both dimensions
        plt.gca().set_aspect('equal', adjustable='box')
        text_x = 0
        text_y = 9
        plt.text(text_x, text_y, f"r2_score: {round(r2, 4)}", fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
        plt.show()

    #Calling the plotting function
    graphPredictions(net, test_loader, -1, 10)
"""