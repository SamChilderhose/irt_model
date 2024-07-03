import sys
sys.path.append("c:\\Users\\samch\\OneDrive\\Documents\\CSC311\\csc311_final\\starter_code")
from utils import *
from student_metadata_loader import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, alpha, theta, beta, gamma, zeta):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    responses = np.array(data['is_correct'], dtype=bool)
    p = logistic(alpha, theta, beta, gamma, zeta, data)

    log_p = np.log(p)
    log_one_minus_p = np.log(1-p)

    log_lklihood = np.sum(responses * log_p + (1 - responses) * log_one_minus_p)
    return -log_lklihood

def update_theta_beta(data, lr, alpha, theta, beta, gamma, zeta):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    p = logistic(alpha, theta, beta, gamma, zeta, data)

    grad_alpha, grad_theta, grad_beta, grad_gamma, grad_zeta = _gradient(p, alpha, theta, beta, gamma, zeta, data)
    
    alpha = alpha - lr * grad_alpha
    theta = theta - lr * grad_theta
    beta = beta - lr * grad_beta
    gamma = gamma - lr * grad_gamma
    zeta = zeta - lr * grad_zeta
    zeta = np.clip(zeta, 0, 1)

    return alpha, theta, beta, gamma, zeta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_students = len(set(data['user_id']))
    num_questions = len(set(data['question_id']))
    alpha =np.random.normal(loc=1.2, scale=0.3, size=num_questions)
    theta = np.random.normal(size=num_students)
    beta = np.random.normal(size=num_questions)
    gamma = np.random.normal(size=num_questions)

    # Zeta is the guessing parameter so it can only be in the range [0,1]
    zeta = np.random.normal(0.2, 0.1, num_questions)
    zeta = np.clip(zeta, 0, 1)

    val_acc_lst = []
    train_acc_lst = []
    val_nllk_lst = []
    train_nllk_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, alpha=alpha, theta=theta, beta=beta, gamma=gamma, zeta=zeta)
        val_neg_lld = neg_log_likelihood(val_data, alpha=alpha, theta=theta, beta=beta, gamma=gamma, zeta=zeta)
        val_score = evaluate(data=val_data, alpha=alpha, theta=theta, beta=beta, gamma=gamma, zeta=zeta)
        train_score = evaluate(data=data, alpha=alpha, theta=theta, beta=beta, gamma=gamma, zeta=zeta)
        
        train_nllk_lst.append(train_neg_lld)
        val_nllk_lst.append(val_neg_lld)
        train_acc_lst.append(val_score)
        val_acc_lst.append(train_score)
        print("Validation -- NLLK: {} \t Score: {}".format(val_neg_lld, val_score))
        print("Training   -- NLLK: {} \t Score: {}".format(train_neg_lld, train_score))

        alpha, theta, beta, gamma, zeta = update_theta_beta(data, lr, alpha, theta, beta, gamma, zeta=zeta)

    return alpha, theta, beta, gamma, zeta, val_acc_lst, train_acc_lst, train_nllk_lst, val_nllk_lst


def evaluate(data, alpha, theta, beta, gamma, zeta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        is_prem = data['is_premium'][u]
        x = (zeta[q] + (1 - zeta[q]) * (alpha[q] * (theta[u] - beta[q] - gamma[q] * is_prem))).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    add_student_meta_data(train_data, val_data, test_data)
    lr = 0.01
    iterations = 50
    alpha, theta, beta, gamma, zeta, val_acc_lst, train_acc_lst, train_nllk_lst, val_nllk_lst = irt(train_data, val_data, lr, iterations)

    make_plots(alpha, theta, beta, val_acc_lst, train_acc_lst, iterations, train_nllk_lst, val_nllk_lst)

    test_accuracy = evaluate(data=test_data, alpha=alpha, theta=theta, beta=beta, gamma=gamma, zeta=zeta)
    print("Final Test Accuracy: " + str(test_accuracy))
    question_j1 = train_data['question_id'][1]
    question_j2 = train_data['question_id'][2]
    question_j3 = train_data['question_id'][3]


    make_probability_plots(alpha, theta, beta, gamma, zeta, train_data, question_j1, question_j2, question_j3)

    print("Discriminant for Question 1030: " + str(alpha[1030]))
    print("Ability for Student 99: " + str(theta[99]))
    print("Difficulty for Question 1030: " + str(beta[1030]))
    print("Premium for Question 1030: " + str(gamma[1030]))
    print("Guessing for Question 1030: " + str(zeta[1030]))

#####################
# Helper Functions: #
#####################

def calculate_probability_list(theta_lst, b, a, g, z, is_prem_lst):
    p_lst = []
    thetas = []
    premium_p_lst = []
    premium_thetas = []

    for i in range(len(theta_lst)):
        t = theta_lst[i]
        is_prem = is_prem_lst[i]

        p = z + (1 - z) / (1 + np.exp(-a * (t - b - g * is_prem)))
        if is_prem == 1:
            premium_thetas.append(t)
            premium_p_lst.append(p)
        else:
            thetas.append(t)
            p_lst.append(p)

    return p_lst, premium_p_lst, thetas, premium_thetas

def logistic(alpha, theta, beta, gamma, zeta, data):
    num_entries = len(data['is_correct'])
    p = np.zeros(shape=num_entries, dtype=float)

    for i in range(num_entries):
        t = theta[data['user_id'][i]]
        b = beta[data['question_id'][i]]
        a = alpha[data['question_id'][i]]
        g = gamma[data['question_id'][i]]
        z = zeta[data['question_id'][i]]
        is_prem = data['is_premium'][data['user_id'][i]]
        
        p[i] = z + ((1 - z) / (1 + np.exp(-a * (t - b - g * is_prem))))

        # if p[i] < 0 or p[i] > 1:
        #     print("Invalid Value of p in logistic: " + str(p[i]))

    return p

def _gradient(p, alpha, theta, beta, gamma, zeta, data) -> list[list,list]:
    theta_grad, beta_grad = np.zeros(shape=theta.shape[0], dtype=float), np.zeros(shape=beta.shape[0], dtype=float)
    alpha_grad = np.zeros(shape=alpha.shape[0], dtype=float)
    gamma_grad = np.zeros(shape=alpha.shape[0], dtype=float)
    zeta_grad = np.zeros(shape=alpha.shape[0], dtype=float)

    for i in range(len(data['is_correct'])):
        ti = data['user_id'][i]
        bj = data['question_id'][i]
        aj = data['question_id'][i]
        gj = data['question_id'][i]
        zj = data['question_id'][i]
        is_prem = data['is_premium'][ti]

        d_alpha = (1 - zeta[zj]) * (theta[ti] - beta[bj] - gamma[gj] * is_prem) * (p[i] - data['is_correct'][i])
        d_theta = (1 - zeta[zj]) * alpha[aj] * (p[i] - data['is_correct'][i])
        d_beta = (1 - zeta[zj]) * alpha[aj] * (data['is_correct'][i] - p[i])
        d_gamma = (1 - zeta[zj]) * is_prem * alpha[aj] * (data['is_correct'][i] - p[i])
        d_zeta = 1 - (data['is_correct'][i] - p[i])

        alpha_grad[aj] += d_alpha
        theta_grad[ti] += d_theta
        beta_grad[bj] += d_beta 
        gamma_grad[bj] += d_gamma 
        zeta_grad[bj] += d_zeta
    
    num_samples = np.zeros(len(set(data['question_id'])))
    for i in range(len(data['is_correct'])):
        bj = data['question_id'][i]
        num_samples[bj] += 1

    for i in range(len(set(data['question_id']))):
        num_questions = num_samples[i]
        zeta_grad[i] = zeta_grad[i] / num_questions


    return alpha_grad, theta_grad, beta_grad, gamma_grad, zeta_grad

def make_plots(alpha, theta, beta, val_acc_lst, train_acc_lst, iterations, train_nllk_lst, val_nllk_lst):
    iter= range(0,iterations)
    plt.plot(iter, val_acc_lst, 'g', label='Training Accuracy')
    plt.plot(iter, train_acc_lst, 'b', label='validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(iter, train_nllk_lst, 'g', label='Training Negative Log Likelihood')
    plt.plot(iter, val_nllk_lst, 'b', label='Validation Negative Log Likelihood')
    plt.title('Train & Validation Negative Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood')
    plt.legend()
    plt.show()

def make_probability_plots(alpha, theta, beta, gamma, zeta, data, question_j1, question_j2, question_j3):
    theta_lst = theta.tolist()
    theta_lst.sort()
    
    is_prem = data['is_premium']

    p1, p1_premium, t1, t1_premium = calculate_probability_list(theta_lst, beta[question_j1], alpha[question_j1], gamma[question_j1], zeta[question_j1], is_prem)
    p2, p2_premium, t2, t2_premium = calculate_probability_list(theta_lst, beta[question_j2], alpha[question_j2], gamma[question_j2], zeta[question_j1], is_prem)
    p3, p3_premium, t3, t3_premium = calculate_probability_list(theta_lst, beta[question_j3], alpha[question_j3], gamma[question_j3], zeta[question_j1], is_prem)

    plt.plot(t1, p1, 'g', label='Question ' + str(question_j1))
    plt.plot(t1_premium, p1_premium, 'g', linestyle='dashed', label='Premium Question ' + str(question_j1))
    plt.plot(t2, p2, 'b', label='Question ' + str(question_j2))
    plt.plot(t2_premium, p2_premium, 'b',  linestyle='dashed', label='Premium Question ' + str(question_j2))
    plt.plot(t3, p3, 'r', label='Question ' + str(question_j3))
    plt.plot(t3_premium, p3_premium, 'r',  linestyle='dashed', label='Premium Question ' + str(question_j3))
    plt.title('Probability Question Answered Correctly vs Student Ability')
    plt.xlabel('Thetas')
    plt.ylabel('Probability Answer Question j Correctly')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
