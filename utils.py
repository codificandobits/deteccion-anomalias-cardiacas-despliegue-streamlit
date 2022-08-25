import pickle, torch

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Codificador
        self.cod1 = torch.nn.Linear(in_features=140, out_features=32) # Entrada: 140 muestras
        self.cod2 = torch.nn.Linear(in_features=32, out_features=16)
        self.cod3 = torch.nn.Linear(in_features=16, out_features=8)
        
        #Decodificador
        self.dec1 = torch.nn.Linear(in_features=8, out_features=16)
        self.dec2 = torch.nn.Linear(in_features=16, out_features=32)
        self.dec3 = torch.nn.Linear(in_features=32, out_features=140)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.cod1(x))
        x = torch.nn.functional.relu(self.cod2(x))
        x = torch.nn.functional.relu(self.cod3(x))
        
        x = torch.nn.functional.relu(self.dec1(x))
        x = torch.nn.functional.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        
        return x

def leer_dato(uploaded_file):
    dato = pickle.loads(uploaded_file.getvalue())

    return dato

def cargar_modelo_preentrenado(model_path):
    modelo = torch.load(model_path)
    modelo.eval()

    return modelo

def predecir(modelo, datos, umbral):
    # Modificaciones:
    # - En lugar de from_numpy -> torch.numpy
    # - En lugar de mean(dim=1) -> mean (pues tendremos sólo 1 dato)
    fn_perdida = torch.nn.L1Loss(reduction='none')
    reconstrucciones = modelo(torch.from_numpy(datos).float())
    perdida = fn_perdida(reconstrucciones, torch.from_numpy(datos).float()).mean()

    return torch.lt(perdida, umbral)

def obtener_categoria(comparaciones):
    # Modificaciones:
    # No se requiere el uso de "for" pues tendremos sólo 1 dato
    if comparaciones.item():
        categoria = 'Normal'
    else:
        categoria = 'Anormal'

    return categoria


