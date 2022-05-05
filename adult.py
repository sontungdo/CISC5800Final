import neuralnet as nn
import numpy as np
import pandas as pd

# read file and pre-process
with open("adult.data", "r") as f:
    lines = f.readlines()

df=[]
for i in lines:
    df.append(i.split(','))
df=df[:-1]
for i in range(len(df)):    
    if (df[i][1] ==' Private'):   ## workclass(1)
        df[i][1]=0
    elif (df[i][1]== ' Never-worked'):
        df[i][1]=2
    else:
        df[i][1]=1

    if (df[i][3] ==' Bachelors'):   ## education(3)
        df[i][3]=1
    elif (df[i][3]== ' Masters' or ' Doctorate'):
        df[i][3]=2
    else:
        df[i][3]=0

    if (df[i][5] ==' Never-married'):   ## marital-status(5)
        df[i][5]=0
    else:
        df[i][5]=1

    if (df[i][8] ==' White'):   ## race(8)
        df[i][8]=0
    elif (df[i][8]== ' Asian-Pac-Islander'):
        df[i][8]=1
    elif (df[i][8]== ' Amer-Indian-Eskimo'):
        df[i][8]=2
    elif (df[i][8]== ' Black'):
        df[i][8]=3
    else:
        df[i][8]=4
    
    if (df[i][9] ==' Male'):   ## sex(9)
        df[i][9]=1
    else:
        df[i][9]=0

    if (df[i][13] ==' United-States'):   ## native-country(13)
        df[i][13]=1
    else:
        df[i][13]=0

    if (df[i][14] ==' >50K\n'):   ## class(14)
        df[i][14]=1
    else:
        df[i][14]=0

for i in df:   ## delete occupation(6) && relationship(7)
    del i[6:8]  
    #del i[2]

df = np.array([[int(j) for j in i] for i in df])

# process imbalanced data
df = pd.DataFrame(df)
df_zero = df[df[12] == 0]
df_one = df[df[12] == 1]

df_zero = df.sample(n=7841)

df = pd.concat([df_zero, df_one])
df = df.sample(frac= 1, random_state = 1)
df = np.array(df)

X = df[:, :-1]
y = df[:,-1]

# train/test split
split = 12000
df_train = df[:split]
df_test = df[split:]

# construct network
model = nn.Network()
input_shape = df.shape[1]-1
model.construct_network(input_shape, [10,2])

# train the model
model.adaptive_train(df_train, n_epoch=10, learning_rate=0.1, batch_size=100)

# test model
model.test_model(df_test)

# for i in range(20):
#     model.forward_propagate(df_test[i,:-1])
#     print(model.predict(df_test[i,:-1]), df_test[i,-1])