import matplotlib.pyplot as plt


def plot_3fold_accuracies(domain, Stats):
    """
        Function to create barplots of the validation and test accuracies resulting of a three fold cross validation for domain only models (No Domain Adapatation). 
    """
    fig = plt.figure(figsize = (7,6))
    plt.bar(['Validation', 'Test'], Stats[0], yerr = Stats[1], capsize = 10)
    plt.title('F1-Score accuracy for Cashew classification in ' + domain + '\n3-fold CV')
    plt.tight_layout()
    fig.savefig(domain + '_accuracy.png', dpi = 200)

def plot_random_test_images(network, test_loader):

    return 0