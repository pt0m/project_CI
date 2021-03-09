# Computational imaging

## dataset :
samoyed-data

## Tests :

- ajout de bruit gaussien sur le dataset puis débruitage et mesure PSNR
- ecart-type bruit gaussien allant de 5% à ????% de l'amplitude de l'image (1 ou 255) donc on aura une courbe
- mean MSE to original images (pareil courbe en fonction de l'ecart type)
- Temps debruitage 1000 images
- *demosaicking ?*

## Methodes que l'on souhaite implémenté:

- Non-local Mean
- Sparse Encoding
- Non-local Sparse Model (the addition of the paper) # plus compliqué

## Non-local Mean

- one group can test the version with a small search area and the other with a searching area that cover the all images


## other

BM3D - pour comparer (a partir d'un github si jamais on trouve et qu'on a fini les autres)
