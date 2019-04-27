import numpy as np

def predict(theta,X):
    """
    Função calculada:
    X - matriz de observações
    theta - variáveis da função calculada
    """
    X = np.asarray(X) ; theta = np.asarray(theta)
    return X.dot(theta) #Calcula o produto matricial entre X e theta

def cost(theta,X,y):
    """
    Função custo que computa os desvios entre o dado observado e o calculado:
    X - Matriz das variáveis observadas (nxm : n - número de variáveis ; m - número de observações)
    y - Vetor de com as respostas esperadas (tamanho m) 
    """
    return np.sum(np.square(predict(theta,X)-y))/len(y)

def gradient_descent(X,y,theta,alpha=0.01,ninter=100):
    
    try: X = np.append(X.reshape((len(X),1)),(np.ones((len(X),1))),axis=1) #Adicionando bias - 1 coluna
    except: X = np.append(X,(np.ones((len(X),1))),axis=1) #Adicionando bias - 2+ coluna

    theta = np.asarray(theta)
    cost_history = np.zeros(ninter) ; theta_history = np.zeros((ninter,2))
    for i in range(ninter):
        theta = theta - alpha*(X.T.dot((predict(theta,X))-y))*(1/len(y))
        theta_history[i,:] = theta.T
        cost_history[i] = cost(theta,X,y)
    return theta, cost_history, theta_history

