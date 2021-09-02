#!/bin/bash
cd ..
docker run --rm --gpus all -v $(pwd)/data:/3DCADFusion/data -v $(pwd)/model:/3DCADFusion/model -it 3dcadfusion

