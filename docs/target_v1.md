geo-locate-ml v1.0 — France-only (Solid)
Objectif produit

À partir d’une photo Mapillary prise en France métropolitaine, prédire une zone (H3 cell) et fournir une estimation de position, avec des erreurs géographiques faibles et stables.

Dataset v1.0
Coverage

50 villes (tes 10 batches actuels)

≥ 100,000 images au total

Mix: grandes métropoles + villes moyennes + 5–10 zones moins denses (éviter “Paris bias”)

Quality constraints

Zero leakage : sequence_id présent dans un seul split ✅ (déjà)

Min samples / cell : ≥ 150 (idéal 200)

Max dominance : la plus grosse classe ≤ 25% du train (sinon ça triche sur la densité)

H3

Base: H3 res 7

Option v1.0+: res 8 si dataset ≥ 150k (sinon trop sparse)

Model v1.0
Architecture

Backbone pretrained (minimum) :

ResNet18 pretrained ImageNet

Head:

Global average pooling

Dropout (0.2–0.4)

Linear → num_classes

Training

Fine-tuning complet

Weight decay + dropout

Data augmentation (rotation légère, color jitter modéré, random crop)

Class weights activés si imbalance

Geo loss activée (ton système actuel)

Metrics v1.0 (Definition of Done)

Tu ships v1.0 si sur test split :

Metric	Target	Gate
Accuracy	≥ 50%	must
Median error	< 1.0 km	must
p90 error	< 5.0 km	must
Stability	variance faible (pas de yoyos)	must

Important : on “gate” sur km, pas sur accuracy.

Inference v1.0

Commande unique :

python -m src.predict path/to/image.jpg

Sortie v1.0 :

top-5 classes (cell id / city label)

lat/lon centroid prédite

distance km vs si GT connu (option)

confiance

Engineering v1.0
Pipeline stable

make batch fonctionne à froid

make export produit un snapshot complet

make clean-raw supprime uniquement les images

Artifacts

runs/latest/REPORT.md (metrics + config)

confusion_matrix.png

geo_error.png

metrics.csv

models/best.pt = global best

Plan d’exécution v1.0 (ordre exact)
Step 1 — Data scale (le vrai levier)

Exécuter batches 01→10 jusqu’à atteindre 100k images

Après chaque batch :

rebuild

sanity

train

export

clean-raw (si nécessaire)

KPI surveillés :

nb images

nb classes (cells)

min/median/max samples per cell

dominance top class

Step 2 — Baseline “pretrained”

Implémenter ResNet18 pretrained

Comparer à ton CNN

Gater sur median/p90 km

Step 3 — Balance & regularization

class weights

augmentation

calibrer dropout/weight_decay

Step 4 — Geo loss tuning

geo_tau_km (typiquement 1.5 → 5.0)

geo_mix_ce (0.25 → 0.55)

Gate sur p90_km

Step 5 — Freeze v1.0

dernier run “gold”

snapshot export

tag Git v1.0