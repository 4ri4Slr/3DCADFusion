mkdir -p ../model
cp -R -u -p ../best_model.pth ../model 2>/dev/null
wget -nc https://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -O ../model/se_resnext50_32x4d-a260b3a4.pth
docker build -t 3dcadfusion ..

