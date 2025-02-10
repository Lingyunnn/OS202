
# TD1

`pandoc -s --toc README.md --css=./github-pandoc.css -o README.html`

## lscpu

*lscpu donne des infos utiles sur le processeur : nb core, taille de cache :*

```
Coller ici les infos *utiles* de lscpu.
```


## Produit matrice-matrice

### Effet de la taille de la matrice

  n            | MFlops
---------------|--------
1024 (origine) |217.901
1023           |315.049
1025           |303.024
1026           |310.718
2048           |166.299

*Expliquer les résultats.*


### Permutation des boucles

*Expliquer comment est compilé le code (ligne de make ou de gcc) : on aura besoin de savoir l'optim, les paramètres, etc. Par exemple :*

`make TestProduct.exe && ./TestProduct.exe 1024`


  ordre           | time    | MFlops(n=1024)  | 
------------------|---------|---------|
i,j,k             | 9.06818 | 236.815 |
j,i,k             | 8.83586 | 243.042 |
i,k,j             | 16.1995 | 132.565 |
k,i,j             | 12.9205 | 166.208 |
j,k,i             | 6.62216 | 324.288 |
k,j,i             | 7.76396 | 276.597 |


*Discuter les résultats.*



### OMP sur la meilleure boucle

`make TestProduct.exe && OMP_NUM_THREADS=8 ./TestProduct.exe 1024`

  OMP_NUM         | MFlops(n=1024)  | MFlops(n=2048) | MFlops(n=512) 
------------------|---------|----------------|----------------
1                 |310.357  |286.935         |327.226         
2                 |606.295  |480.046         |607.964
3                 |930.962  |607.799         |872.122
4                 |1153.37  |780.999         |1194.44
5                 |1304.31  |1307.78         |1407.85
6                 |1583.49  |1432.76         |1503.11
7                 |1617.94  |1584.08         |1631.47
8                 |1763.61  |1629.57         |1746.88

*Tracer les courbes de speedup (pour chaque valeur de n), discuter les résultats.*



### Produit par blocs

`make TestProduct.exe && ./TestProduct.exe 1024`

  szBlock         | MFlops(n=1024)  | MFlops(n=2048) | MFlops(n=512)  
------------------|---------|----------------|----------------|
origine (=max)    |310.357  |286.935         |327.226         |
32                |284.626  |284.954         |283.954         |
64                |296.135  |273.646         |284.238         |
128               |306.132  |303.171         |304.021         |
256               |312.951  |295.624         |311.288         |
512               |303.063  |306.071         |302.399         |
*Discuter les résultats.*



### Bloc + OMP


  szBlock      | OMP_NUM | MFlops(n=1024)  | MFlops(n=2048) | MFlops(n=512)  |
---------------|---------|---------|----------------|----------------|
1024           |  1      |300.225  |300.70          |303.712         |               
1024           |  8      |1747.33  |1682.98         |1734.67         |               
512            |  1      |303.063  |306.071         |302.399         |                
512            |  8      |1774.72  |1674.68         |1730.11         |   
256            |  1      |312.951  |295.624         |311.288         |            
256            |  8      |1811.02  |1758.67         |1818.44         |
*Discuter les résultats.*


### Comparaison avec BLAS, Eigen et numpy

*Comparer les performances avec un calcul similaire utilisant les bibliothèques d'algèbre linéaire BLAS, Eigen et/ou numpy.*


# Tips

```
	env
	OMP_NUM_THREADS=4 ./produitMatriceMatrice.exe
```

```
    $ for i in $(seq 1 4); do elap=$(OMP_NUM_THREADS=$i ./TestProductOmp.exe|grep "Temps CPU"|cut -d " " -f 7); echo -e "$i\t$elap"; done > timers.out
```
