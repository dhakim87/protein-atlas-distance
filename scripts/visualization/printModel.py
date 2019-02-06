
from keras.applications import resnet50
from keras.utils import plot_model

#model = resnet50.ResNet50(weights=None)  #Untrained
model = resnet50.ResNet50(weights="imagenet")  #Pre trained on ImageNet

#Print the model to a file.
plot_model(model, to_file='model.svg');

#print resnet_model.summary();
with open("model_config.txt", "w") as f:
    for layer in model.layers:
        f.write(layer.name)
        f.write(" ")
        f.write(str(layer.get_config()))
        f.write("\n")

model.summary();
