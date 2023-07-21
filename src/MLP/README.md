The following are the results obtained on evaluation the modle on our corpus

Activation function: relu       l2_norm: 0.001  Accuracy: 69.9%
Activation function: relu       l2_norm: 0.01   Accuracy: 68.56
Activation function: relu       l2_norm: 0.1    Accuracy: 68.73
Activation function: sigmoid    l2_norm: 0.001  Accuracy: 72.96
Activation function: sigmoid    l2_norm: 0.01   Accuracy: 66.78
Activation function: sigmoid    l2_norm: 0.1    Accuracy: 71.16
Activation function: tanh       l2_norm: 0.001  Accuracy: 69.65
Activation function: tanh       l2_norm: 0.01   Accuracy: 69.34
Activation function: tanh       l2_norm: 0.1    Accuracy: 69.37

The best model is the one with sigmoid as its activation function and l2_normalization value as 0.001

Activation functions, L2-norm regularization, and dropout are all techniques commonly used in neural networks to improve model performance and generalization.

Activation Functions:

Relu and tanh activation functions give almost the same accuracy. Whereas sigmoid activation function has better results. This is because the output of sigmoid activation function is between 0 and 1. Also we will not be facing vanishing gradient or exploding gradient issues as out neural network is shallow. 

Activation functions introduce non-linearity to neural networks, enabling them to learn complex patterns and relationships in the data. Different activation functions have varying properties, such as differentiable behavior, range of output values, and ability to mitigate the vanishing gradient problem.
Activation functions like ReLU (Rectified Linear Unit) have been widely adopted due to their simplicity and effectiveness. ReLU helps address the vanishing gradient problem by allowing the network to propagate gradients for positive inputs.
Choosing the right activation function is crucial as it can impact model capacity, convergence speed, and the ability to learn complex features. Improper selection may lead to underfitting or overfitting.


L2-Norm Regularization:

L2-norm regularization, also known as weight decay, helps prevent overfitting by adding a penalty term to the loss function that discourages large weight values. It encourages the model to favor smaller weights and, in turn, reduces the complexity of the learned function.
L2 regularization can improve generalization by reducing the model's reliance on specific features and making it less sensitive to noise or outliers in the training data. It can help control the trade-off between fitting the training data and avoiding overfitting.
The effect of L2-norm regularization can be seen in the model's weights, where the regularization term pushes them towards smaller values. This can lead to smoother decision boundaries and more robust predictions.


Dropout:

Dropout is a regularization technique where randomly selected neurons are temporarily dropped out or ignored during training. This means that their outputs and connections are set to zero with a certain probability.
Dropout acts as a form of ensemble learning by training multiple sub-networks within the main network. It helps prevent overfitting by reducing the interdependencies between neurons and encouraging the network to learn more robust and generalized representations.
Dropout also acts as a regularizer by implicitly creating a noise injection mechanism, making the model more resilient to small changes in the input and reducing the reliance on specific features.
Intuitively, dropout can be seen as forcing the network to be more self-reliant and not overly dependent on any particular set of neurons, thereby promoting better generalization.
Overall, activation functions determine the non-linear behavior of the neural network, regularization techniques like L2-norm help control model complexity and prevent overfitting, and dropout promotes robustness and generalization by reducing interdependencies between neurons. These techniques, when applied appropriately, can improve model performance, prevent overfitting, and enhance the model's ability to generalize to unseen data.