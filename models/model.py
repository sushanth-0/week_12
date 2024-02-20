import pickle


def load_model():
    model = pickle.load(open('model_1.pkl', 'rb'))
    return model
