import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


#-----------------------------------------------------------------------------------------------------------------------
# charger les données
#-----------------------------------------------------------------------------------------------------------------------
def import_data(path='./data/dataset.txt', sep= '\t'):
    """
    Importer les données
    :param path: le chemin vers les données
    :param sep: le séprateur utilisé pour lire les données
    :return: np matrice et df des données
    """
    data = pd.read_csv(path, sep)
    #
    # éliminer les colonnes que nous n'utiliserons pas
    try:
        mydata = data.drop(['Points', 'Rank', 'Competition'], axis=1).fillna(0)
        #
        # transformer les données en array numpy
        X = mydata.values
    except:
        try:
            mydata = data.drop(['idCours', 'titreCours'], axis=1).fillna(0)
            # transformer les données en array numpy
            X = mydata.values
        except:
            X = data.values
    #
    return X, data

#-----------------------------------------------------------------------------------------------------------------------
# Standardiser les données en entrée
#-----------------------------------------------------------------------------------------------------------------------
def standardiser(X):
    """
    Permet de standardiser les données
    :param X: np matrice des données
    :return: np matrice standardisé
    """
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)
    return X_scaled

#-----------------------------------------------------------------------------------------------------------------------
# Calcul des composantes principales
#-----------------------------------------------------------------------------------------------------------------------
def pratiquer_pca(X_scaled, nb_componenys):
    """
    Réaliser la pca.
    :param X_scaled: np matrice des données
    :param nb_componenys: nb de composentes principales de l'acp
    :return: pca model and new pca components
    """
    if nb_componenys == -1:
        pca = decomposition.PCA()
    else:
        pca = decomposition.PCA(n_components=nb_componenys)
    pca.fit(X_scaled)
    # projeter X sur les composantes principales
    X_projected = pca.transform(X_scaled)
    return pca, X_projected


#-----------------------------------------------------------------------------------------------------------------------
# Pourcentage de variance expliquée
#-----------------------------------------------------------------------------------------------------------------------
def compute_explained_variance(pca):
    """
    Compute and returned explained variance
    :param pca: the pca model
    :return: Explained variance and cumulative explmained variance
    """
    explained_variance = pca.explained_variance_ratio_
    cummulative_explained_variance = pca.explained_variance_ratio_.cumsum()
    print(explained_variance)
    print(cummulative_explained_variance)
    return explained_variance, cummulative_explained_variance
# La première composante explique environ un tiers de la variance observée dans les données, et la deuxième 17.3 %.
# Au total, ces deux composantes expliquent 50 % de la variance totale, en utilisant seulement un cinquième des
# dimensions initiales.
# Nous pouvons représenter chaque athlète/compétition selon ces deux dimensions uniquement, et colorer chacun des points
# correspondants en fonction du classement de l'athlète lors de cette compétition.


def plot_observation(X_projected, data):
    # afficher chaque observation
    plt.scatter(X_projected[:, 0], X_projected[:, 1],
                # colorer en utilisant la variable 'Rank'
                c=data.Rank)
    plt.xlim([-5.5, 5.5])
    plt.ylim([-4, 4])
    plt.colorbar()
    plt.show()
# Les bonnes performances (points bleu foncé) sont plutôt situées dans la partie droite du graphe (PC1 > 0) et les moins
# bonnes (points jaunes) plutôt dans la partie gauche (PC1 < 0).

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

#-----------------------------------------------------------------------------------------------------------------------
# Contribution de chaque variable aux composantes principales
#-----------------------------------------------------------------------------------------------------------------------
def plot_correlation_cercle(pca, data):
    pcs = pca.components_
    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        # Afficher un segment de l'origine au point (x, y)
        plt.plot([0, x], [0, y], color='k')
        # Afficher le nom (data.columns[i]) de la performance
        plt.text(x, y, data.columns[i], fontsize='14')
    #
    # Afficher une ligne horizontale y=0
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
    #
    # Afficher une ligne verticale x=0
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
    #
    plt.xlim([-0.7, 0.7])
    plt.ylim([-0.7, 0.7])
    plt.show()

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            #
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))
            #
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
            #
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1,:], pcs[d2,:],
                           angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            #
            # affichage des noms des variables
            if labels is not None:
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            #
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)
            #
            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            #
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
            #
            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            #
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            #
            # initialisation de la figure
            fig = plt.figure(figsize=(7,6))
            #
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()
            #
            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center',va='center')
            #
                    # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
            #
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')
            #
            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            #
            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

if __name__ == '__main__':
    X, data = import_data(path='./data/my_courses.csv', sep=',')
    X_scaled = standardiser(X)
    pca, X_projected = pratiquer_pca(X_scaled, nb_componenys=-1)
    explained_variance, explained_cumul_variance = compute_explained_variance(pca)
    # On peut choisir 5 composentes principales

    n_comp = 5
    pca, X_projected = pratiquer_pca(X_scaled, nb_componenys=n_comp)
    explained_variance, explained_cumul_variance = compute_explained_variance(pca)
    plot_correlation_cercle(pca, data)
    # Eboulis des valeurs propres
    display_scree_plot(pca)

    # Cercle des corrélations
    pcs = pca.components_
    display_circles(pcs, n_comp, pca, [(0, 1), (2, 3), (4, 5)], labels=np.array(data.columns))

    display_factorial_planes(X_projected, n_comp, pca, [(0, 1), (2, 3), (4, 5)], labels=np.array(data.idCours))
    display_factorial_planes(X_projected, n_comp, pca, [(0, 1), (2, 3), (4, 5)], labels=np.array(data.titreCours))
